"""Live forecast: pull the latest data, predict the next 72 hours, write
outputs/latest.json for the dashboard, and log forecasts so their accuracy can
be checked later against what actually happened (outputs/live_skill.json).

Run with:  python -m smogcast forecast
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import aqi, config
from .clean import clean_pm25
from .data import firms, openaq, weather
from .features import build_panel, make_rows
from .train import MAX_H, predict

LOG = config.OUTPUT_DIR / "forecast_log.csv.gz"
IST = "Asia/Kolkata"
LO, HI = (f"q{int(q * 100)}" for q in config.QUANTILES)


def fetch_live(stations: pd.DataFrame, days: int = 14):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    start = (now - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    pm, wx = [], []
    for _, s in stations.iterrows():
        try:
            pm.append(openaq.fetch_sensor_hours(s.sensor_id, start, now.strftime("%Y-%m-%dT%H:%M:%SZ")))
        except Exception as e:
            print(f"  PM2.5 fetch failed for {s.station}: {e}")
        w = weather.fetch_forecast(s.lat, s.lon, past_days=days, forecast_days=4)
        w["sensor_id"] = s.sensor_id
        wx.append(w)
    fires = firms.to_events(firms.fetch_recent(days=5))
    pm = pd.concat(pm, ignore_index=True) if pm else pd.DataFrame(columns=["time", "pm25", "sensor_id"])
    return pm, pd.concat(wx, ignore_index=True), fires


def make_forecast(bundle: dict, pm_raw: pd.DataFrame, wx: pd.DataFrame, fires: pd.DataFrame,
                  now: pd.Timestamp | None = None) -> dict:
    stations, feats, models = bundle["stations"], bundle["features"], bundle["models"]
    now = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    pm = clean_pm25(pm_raw)

    # issue time = latest hour where at least half the stations reported
    counts = pm[pm["pm25_obs"]].groupby("time").size()
    ok = counts[counts >= max(1, len(stations) // 2)]
    if ok.empty:
        raise SystemExit("No recent PM2.5 data; cannot issue a forecast.")
    issue = ok.index.max()

    index = pd.date_range(issue - pd.Timedelta(days=9), issue + pd.Timedelta(hours=MAX_H), freq="h", tz="UTC")
    panel = build_panel(pm, stations, wx, fires, index=index)
    rows = make_rows(panel, range(1, MAX_H + 1), issue_mask=lambda g: g["time"] == issue, with_target=False)
    rows = rows.dropna(subset=["pm_log_now"])
    # keep targets up to 72 h after *now* (the latest reading may be a few days old)
    rows = rows[pd.to_datetime(rows["target_time"], utc=True) <= now + pd.Timedelta(hours=72)].reset_index(drop=True)
    if rows.empty:
        raise SystemExit("No station has a reading at the issue time.")

    pred = predict(models, rows[feats])
    X0 = rows[feats].copy()
    X0[[c for c in feats if c.startswith("fire_")]] = 0.0
    no_fire = np.expm1(models["mean"].predict(X0)).clip(min=0)
    rows = pd.concat([rows[["sensor_id", "time", "target_time", "horizon"]], pred], axis=1)
    rows["fire"] = (rows["mean"] - no_fire).clip(lower=0)

    out = {"issued_at": issue.isoformat(), "generated_at": now.isoformat(),
           "data_lag_hours": int((now - issue) / pd.Timedelta(hours=1)),
           "model_trained_at": bundle.get("trained_at"), "data": bundle.get("data", "real"),
           "stations": []}
    for _, st in stations.iterrows():
        r = rows[rows["sensor_id"] == st.sensor_id].sort_values("horizon")
        if r.empty:
            continue
        hist = pm[(pm["sensor_id"] == st.sensor_id) & (pm["time"] > issue - pd.Timedelta(hours=72))]
        tt = pd.to_datetime(r["target_time"], utc=True)
        r = r.assign(date=tt.dt.tz_convert(IST).dt.date)
        daily = []
        for d, g in r.groupby("date"):
            if len(g) < 12 or d < now.tz_convert(IST).date():
                continue
            m = float(g["mean"].mean())
            cat = aqi.category(m)
            daily.append({"date": str(d), "pm25": round(m, 1), "aqi": round(aqi.sub_index(m)),
                          "category": cat, "advice": aqi.ADVICE[cat],
                          "fire_pm25": round(float(g["fire"].mean()), 1)})
        out["stations"].append({
            "sensor_id": int(st.sensor_id), "name": st.station,
            "lat": float(st.lat), "lon": float(st.lon),
            "latest_pm25": round(float(hist["pm25"].dropna().iloc[-1]), 1) if hist["pm25"].notna().any() else None,
            "history": [{"time": t.isoformat(), "pm25": None if np.isnan(v) else round(float(v), 1)}
                        for t, v in zip(hist["time"], hist["pm25"])],
            "forecast": [{"time": t.isoformat(), "pm25": round(float(a), 1), "lo": round(float(b), 1),
                          "hi": round(float(c), 1), "fire": round(float(f), 1)}
                         for t, a, b, c, f in zip(tt, r["mean"], r[LO], r[HI], r["fire"])],
            "daily": daily,
        })
    recent = fires[pd.to_datetime(fires["time"], utc=True) > issue - pd.Timedelta(hours=48)]
    out["fires_48h"] = {"count": int(len(recent)), "total_frp": round(float(recent["frp"].sum()), 1),
                        "points": recent.sample(min(len(recent), 1500), random_state=0)[["lat", "lon", "frp"]]
                        .round(3).to_dict("records")}

    _append_log(rows, issue)
    return out


def _append_log(rows: pd.DataFrame, issue: pd.Timestamp) -> None:
    log = rows[["sensor_id", "target_time", "horizon", "mean"]].rename(columns={"mean": "pred"})
    log.insert(0, "issued_at", issue)
    if LOG.exists():
        old = pd.read_csv(LOG)
        old = old[old["issued_at"] != str(issue)]
        log = pd.concat([old, log.astype({"issued_at": str, "target_time": str})], ignore_index=True)
    log.to_csv(LOG, index=False)


def verify(pm_raw: pd.DataFrame) -> dict | None:
    """Score past live forecasts against observations that have since arrived."""
    if not LOG.exists():
        return None
    log = pd.read_csv(LOG)
    log["target_time"] = pd.to_datetime(log["target_time"], utc=True, format="ISO8601")
    pm = clean_pm25(pm_raw)
    pm = pm[pm["pm25_obs"]][["sensor_id", "time", "pm25"]]
    j = log.merge(pm, left_on=["sensor_id", "target_time"], right_on=["sensor_id", "time"])
    if j.empty:
        return None
    j["err"] = (j["pred"] - j["pm25"]).abs()
    j["bucket"] = pd.cut(j["horizon"], [0, 24, 48, 72, MAX_H], labels=["Day 1", "Day 2", "Day 3", "Days 4-6"])
    res = {"n": int(len(j)), "MAE_by_day": j.groupby("bucket", observed=True)["err"].mean().round(1).to_dict(),
           "from": str(j["issued_at"].min()), "to": str(j["issued_at"].max())}
    (config.OUTPUT_DIR / "live_skill.json").write_text(json.dumps(res, indent=2))
    return res


def run() -> None:
    bundle = joblib.load(config.MODEL_DIR / "smogcast.joblib")
    print("Fetching latest PM2.5, weather forecast and fires ...")
    pm, wx, fires = fetch_live(bundle["stations"])
    out = make_forecast(bundle, pm, wx, fires)
    (config.OUTPUT_DIR / "latest.json").write_text(json.dumps(out, indent=1))
    skill = verify(pm)
    print(f"Forecast issued at {out['issued_at']} for {len(out['stations'])} stations.")
    if skill:
        print(f"Live skill so far: {skill['MAE_by_day']}")
