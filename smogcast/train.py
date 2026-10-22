"""Training + honest evaluation.

1. Build features for every station-hour.
2. Train on everything before TEST_START (with a 3-day gap so no target leaks
   across the split) and evaluate on the held-out winter season.
3. Compare against baselines: persistence, 24-h mean persistence, climatology.
4. Estimate the fire contribution with a counterfactual (fire features zeroed).
5. Refit on all data and save the model used for live forecasts.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import aqi, config
from .features import build_panel, feature_columns, make_rows

# Up to 144 h: CPCB data on OpenAQ can arrive 2-3 days late, so a "72 h ahead of now"
# forecast may need up to 144 h ahead of the latest reading.
TRAIN_HORIZONS = [1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60, 66, 72,
                  84, 96, 108, 120, 132, 144]
MAX_H = 144
EVAL_ISSUE_HOUR_UTC = 18           # 23:30 IST: "tonight's forecast for tomorrow"
GROUPS = {
    "PM2.5 history": ("pm_",),
    "Weather now": ("met_",),
    "Weather forecast (target hour)": ("tgt_met_",),
    "Crop fires": ("fire_",),
    "Calendar": ("cal_",),
}


def _model(loss="squared_error", quantile=None, max_iter=500):
    kw = dict(loss=loss, learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=63,
              min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
              random_state=0)
    if quantile is not None:
        kw["quantile"] = quantile
    return HistGradientBoostingRegressor(**kw)


def fit_models(X: pd.DataFrame, y: np.ndarray, max_iter=500) -> dict:
    cat = np.array([c == "station_code" for c in X.columns])
    models = {}
    m = _model(max_iter=max_iter)
    m.set_params(categorical_features=cat)
    models["mean"] = m.fit(X, y)
    for q in config.QUANTILES:
        mq = _model("quantile", q, max_iter=max_iter // 2)
        mq.set_params(categorical_features=cat)
        models[f"q{int(q * 100)}"] = mq.fit(X, y)
    return models


def predict(models: dict, X: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=X.index)
    for k, m in models.items():
        out[k] = np.expm1(m.predict(X)).clip(min=0)
    # keep the interval ordered and around the point forecast
    lo, hi = f"q{int(config.QUANTILES[0] * 100)}", f"q{int(config.QUANTILES[1] * 100)}"
    out[lo] = np.minimum(out[lo], out["mean"])
    out[hi] = np.maximum(out[hi], out["mean"])
    return out


def _mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m])))


def _rmse(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def evaluate(models, ev: pd.DataFrame, feats: list[str], clim: pd.Series) -> dict:
    pred = predict(models, ev[feats])
    y = np.expm1(ev["y"].values)
    base = {
        "Persistence (value now)": np.expm1(ev["pm_log_now"].values),
        "Persistence (last 24h mean)": np.expm1(ev["pm_log_mean24"].values),
    }
    key = pd.MultiIndex.from_arrays([ev["station_code"], ev["cal_month"], ev["cal_hour"]])
    base["Climatology (station x month x hour)"] = clim.reindex(key).values

    res = {"hourly": {}, "daily_mean": {}}
    buckets = {"Day 1 (1-24h)": (1, 24), "Day 2 (25-48h)": (25, 48), "Day 3 (49-72h)": (49, 72),
               "Days 4-6 (73-144h)": (73, 144)}
    for name, (a, b) in buckets.items():
        m = ev["horizon"].between(a, b).values
        row = {"SmogCast": {"MAE": _mae(pred["mean"].values[m], y[m]),
                            "RMSE": _rmse(pred["mean"].values[m], y[m])}}
        for bn, bv in base.items():
            row[bn] = {"MAE": _mae(bv[m], y[m]), "RMSE": _rmse(bv[m], y[m])}
        lo, hi = pred.iloc[:, 1].values[m], pred.iloc[:, 2].values[m]
        ok = ~np.isnan(y[m])
        row["SmogCast"]["interval_coverage"] = float(np.mean((y[m][ok] >= lo[ok]) & (y[m][ok] <= hi[ok])))
        res["hourly"][name] = row

    # daily-mean (what the AQI is based on) and AQI category accuracy
    df = ev[["sensor_id", "time", "horizon"]].copy()
    df["y"], df["pred"] = y, pred["mean"].values
    for bn, bv in base.items():
        df[bn] = bv
    df["day"] = ((df["horizon"] - 1) // 24) + 1
    daily = df.groupby(["sensor_id", "time", "day"]).mean(numeric_only=True)
    daily = daily[df.groupby(["sensor_id", "time", "day"])["y"].count() >= 18]
    for d in range(1, MAX_H // 24 + 1):
        g = daily.xs(d, level="day")
        cat_true = g["y"].map(aqi.category)
        row = {"SmogCast": {"MAE": _mae(g["pred"].values, g["y"].values),
                            "category_accuracy": float((g["pred"].map(aqi.category) == cat_true).mean())}}
        for bn in base:
            row[bn] = {"MAE": _mae(g[bn].values, g["y"].values),
                       "category_accuracy": float((g[bn].map(aqi.category) == cat_true).mean())}
        res["daily_mean"][f"Day {d}"] = row
    res["n_forecasts"] = int(daily.shape[0])

    # the part of the year that matters most: the Oct-Jan smog season
    months = pd.to_datetime(daily.index.get_level_values("time"), utc=True).tz_convert("Asia/Kolkata").month
    smog = daily[np.isin(months, [10, 11, 12, 1])]
    res["daily_mean_smog_season"] = {}
    if len(smog):
        for d in (1, 2, 3):
            g = smog.xs(d, level="day")
            cat_true = g["y"].map(aqi.category)
            row = {"SmogCast": {"MAE": _mae(g["pred"].values, g["y"].values),
                                "category_accuracy": float((g["pred"].map(aqi.category) == cat_true).mean())}}
            for bn in base:
                row[bn] = {"MAE": _mae(g[bn].values, g["y"].values),
                           "category_accuracy": float((g[bn].map(aqi.category) == cat_true).mean())}
            res["daily_mean_smog_season"][f"Day {d}"] = row
        res["n_forecasts_smog_season"] = int(len(smog.xs(1, level="day")))
    return res


def fire_counterfactual(models, ev: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Prediction with real fire features minus prediction with fires set to zero.
    This is the model's estimate of the smoke contribution, not a causal measurement."""
    X = ev[feats]
    X0 = X.copy()
    for c in feats:
        if c.startswith("fire_"):
            X0[c] = 0.0
    p = np.expm1(models["mean"].predict(X)).clip(min=0)
    p0 = np.expm1(models["mean"].predict(X0)).clip(min=0)
    out = pd.DataFrame({"target_time": ev["target_time"].values, "sensor_id": ev["sensor_id"].values,
                        "pred": p, "pred_no_fire": p0})
    out["fire_contribution"] = (out["pred"] - out["pred_no_fire"]).clip(lower=0)
    out["date"] = pd.to_datetime(out["target_time"], utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    daily = out.groupby("date")[["pred", "pred_no_fire", "fire_contribution"]].mean()
    daily["fire_share"] = (daily["fire_contribution"] / daily["pred"]).fillna(0)
    return daily.reset_index()


def group_importance(models, ev: pd.DataFrame, feats: list[str], n=20000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    s = ev.sample(min(n, len(ev)), random_state=seed)
    X, y = s[feats].copy(), np.expm1(s["y"].values)
    base = _mae(np.expm1(models["mean"].predict(X)), y)
    out = {}
    for g, prefixes in GROUPS.items():
        cols = [c for c in feats if c.startswith(prefixes)]
        Xp = X.copy()
        perm = rng.permutation(len(Xp))
        Xp[cols] = Xp[cols].values[perm]
        out[g] = _mae(np.expm1(models["mean"].predict(Xp)), y) - base
    return {"baseline_MAE": base, "MAE_increase_when_shuffled": out}


def run(stations, pm, weather, fires, issue_step=3, test_start=config.TEST_START,
        max_iter=500, final_refit=True, tag="real") -> dict:
    print("Building feature panel ...")
    panel = build_panel(pm, stations, weather, fires)

    print("Building training rows ...")
    rows = make_rows(panel, TRAIN_HORIZONS, issue_mask=lambda g: g["time"].dt.hour % issue_step == 0)
    rows = rows.dropna(subset=["y", "pm_log_now"])
    feats = feature_columns(rows)
    t0 = pd.Timestamp(test_start, tz="UTC")
    train = rows[pd.to_datetime(rows["target_time"], utc=True) < t0 - pd.Timedelta(hours=MAX_H)]
    print(f"  {len(train):,} training rows, {len(feats)} features")

    print("Training ...")
    models = fit_models(train[feats], train["y"].values, max_iter=max_iter)

    # climatology baseline from training period only
    tr_pm = np.expm1(train["y"])
    clim = tr_pm.groupby([train["station_code"], train["cal_month"], train["cal_hour"]]).mean()

    print("Evaluating on held-out season ...")
    ev = make_rows(panel, range(1, MAX_H + 1),
                   issue_mask=lambda g: (g["time"] >= t0) & (g["time"].dt.hour == EVAL_ISSUE_HOUR_UTC))
    ev = ev.dropna(subset=["y", "pm_log_now"])
    metrics = evaluate(models, ev, feats, clim)
    metrics.update({
        "data": tag,
        "train_period": [str(train["time"].min()), str(train["target_time"].max())],
        "test_period": [str(ev["time"].min()), str(ev["target_time"].max())],
        "n_train_rows": int(len(train)), "n_stations": int(stations.shape[0]),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    })
    metrics["importance"] = group_importance(models, ev, feats)
    fire = fire_counterfactual(models, ev, feats)

    (config.OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    fire.to_csv(config.OUTPUT_DIR / "fire_contribution.csv", index=False)
    _print_summary(metrics)

    if final_refit:
        print("Refitting on all data for deployment ...")
        models = fit_models(rows[feats], rows["y"].values, max_iter=max_iter)
    joblib.dump({"models": models, "features": feats, "stations": stations.reset_index(drop=True),
                 "trained_at": metrics["trained_at"], "data": tag},
                config.MODEL_DIR / "smogcast.joblib", compress=3)
    print(f"Saved model to {config.MODEL_DIR / 'smogcast.joblib'}")
    return metrics


def _print_summary(m: dict) -> None:
    print("\nNext-day (24h mean) PM2.5, held-out period:")
    for name, v in m["daily_mean"]["Day 1"].items():
        print(f"  {name:40s} MAE {v['MAE']:6.1f} ug/m3   AQI category acc {v['category_accuracy']:.0%}")
    if m.get("daily_mean_smog_season"):
        print("Same, smog season only (Oct-Jan):")
        for name, v in m["daily_mean_smog_season"]["Day 1"].items():
            print(f"  {name:40s} MAE {v['MAE']:6.1f} ug/m3   AQI category acc {v['category_accuracy']:.0%}")
    print("Group importance (MAE increase when shuffled):")
    for g, v in sorted(m["importance"]["MAE_increase_when_shuffled"].items(), key=lambda x: -x[1]):
        print(f"  {g:35s} +{v:.1f}")
