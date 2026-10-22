"""Data collection: builds data/raw/* from the three APIs.

Each sensor's history is cached in its own file, so an interrupted run resumes
where it stopped. Run with:  python -m smogcast fetch
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config
from .clean import clean_pm25, coverage
from .data import firms, openaq, weather


def save(df: pd.DataFrame, name: str, folder: Path = config.RAW_DIR) -> Path:
    path = folder / f"{name}.csv.gz"
    df.to_csv(path, index=False)
    return path


def load(name: str, folder: Path = config.RAW_DIR) -> pd.DataFrame:
    path = folder / f"{name}.csv.gz"
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df


def _yearly_chunks(start: str, end: str):
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s <= e:
        nxt = min(s + pd.DateOffset(years=1) - pd.Timedelta(days=1), e)
        yield s.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        s = nxt + pd.Timedelta(days=1)


def _monthly_chunks(start: str, end: str):
    """[month_start, next_month_start) pairs as ISO strings; one month is <1000 hourly rows,
    so each request is a single small page (big ranges make OpenAQ time out)."""
    s = pd.Timestamp(start).to_period("M").to_timestamp()
    e = pd.Timestamp(end) + pd.Timedelta(days=1)
    while s < e:
        n = s + pd.DateOffset(months=1)
        yield s.strftime("%Y-%m-%dT00:00:00Z"), min(n, e).strftime("%Y-%m-%dT00:00:00Z")
        s = n


def fetch_history(start: str = config.HISTORY_START, end: str | None = None) -> None:
    end = end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache = config.RAW_DIR / "sensors"
    cache.mkdir(exist_ok=True)

    # 1. stations
    print("Finding PM2.5 stations in Delhi NCR ...")
    st = openaq.find_stations()
    if st.empty:
        raise SystemExit("No stations returned by OpenAQ. Check the API key and bbox.")
    print(f"  {len(st)} candidate sensors")

    # 2. cheap metadata call per sensor. OpenAQ re-ingested CPCB under new sensor IDs in
    #    Feb 2025, so a station's history is split across an old sensor (to ~Oct 2022) and
    #    a new one (Feb 2025 onward). We stitch all of a location's sensors together.
    info_f = config.RAW_DIR / "sensor_info.csv"
    if info_f.exists():
        info = pd.read_csv(info_f)
    else:
        rows = []
        for _, row in st.iterrows():
            try:
                rows.append({"sensor_id": row.sensor_id, **openaq.sensor_info(row.sensor_id)})
            except Exception as e:
                print(f"    metadata failed for sensor {row.sensor_id}: {e}")
        info = pd.DataFrame(rows)
        info.to_csv(info_f, index=False)
    st = st.merge(info, on="sensor_id", how="inner")
    st["first"] = pd.to_datetime(st["first"], utc=True, errors="coerce", format="ISO8601")
    st["last"] = pd.to_datetime(st["last"], utc=True, errors="coerce", format="ISO8601")
    t0, t1 = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    st["overlap_days"] = ((st["last"].clip(upper=t1) - st["first"].clip(lower=t0)).dt.days).clip(lower=0)

    recent = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=14)
    locs = []
    for loc_id, g in st.groupby("location_id"):
        live = g[g["last"] > recent].sort_values("last", ascending=False)
        if live.empty:
            continue                                      # station no longer reporting
        locs.append({"location_id": loc_id, "station": live.iloc[0]["station"],
                     "provider": live.iloc[0]["provider"],
                     "lat": live.iloc[0]["lat"], "lon": live.iloc[0]["lon"],
                     "sensor_id": live.iloc[0]["sensor_id"],   # key used for live forecasts
                     "all_sensors": ";".join(str(x) for x in g["sensor_id"]),
                     "history_days": int(g["overlap_days"].sum())})
    locs = (pd.DataFrame(locs).sort_values("history_days", ascending=False)
            .head(config.MAX_STATIONS).reset_index(drop=True))
    print(f"  {len(locs)} stations selected (most history first): "
          f"{locs['history_days'].min()}-{locs['history_days'].max()} days each")
    save(locs, "stations_all")

    # 3. PM2.5, one month per request, all sensors of each station stitched together
    months = list(_monthly_chunks(start, end))
    by_sid = st.set_index("sensor_id")
    frames = []
    for i, loc in locs.iterrows():
        f = cache / f"pm25_loc{loc.location_id}_{start}.csv.gz"
        if f.exists():
            df = pd.read_csv(f)
            df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
            frames.append(df)
            continue
        sids = [int(x) for x in loc["all_sensors"].split(";")]
        todo = [(sid, a, b) for sid in sids for a, b in months
                if pd.Timestamp(b) > by_sid.loc[sid, "first"] and pd.Timestamp(a) <= by_sid.loc[sid, "last"]]
        print(f"  [{i + 1}/{len(locs)}] {loc.station}: {len(todo)} monthly requests")
        parts, fails = [], 0
        for sid, a, b in todo:
            try:
                parts.append(openaq.fetch_sensor_hours(sid, a, b))
            except Exception as e:
                fails += 1
                print(f"    sensor {sid} {a[:7]} failed: {e}")
                if fails >= 4:
                    print("    too many failures; skipping this station")
                    parts = []
                    break
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        df["sensor_id"] = loc["sensor_id"]             # one id per station from here on
        df.to_csv(f, index=False)
        frames.append(df)
    if not frames:
        raise SystemExit("No PM2.5 data downloaded. See the errors above.")
    pm_raw = pd.concat(frames, ignore_index=True)
    pm_raw["time"] = pd.to_datetime(pm_raw["time"], utc=True, format="ISO8601")
    save(pm_raw, "pm25_raw")

    pm = clean_pm25(pm_raw)
    cov = coverage(pm, start, end)
    keep = cov[cov >= config.MIN_COVERAGE].sort_values(ascending=False)
    if len(keep) < 5:
        keep = cov.sort_values(ascending=False).head(config.MAX_STATIONS)
    stations = locs[locs["sensor_id"].isin(keep.index)].copy()
    stations["coverage"] = stations["sensor_id"].map(keep).round(3)
    save(stations, "stations")
    print(f"Kept {len(stations)} stations (coverage {stations['coverage'].min():.0%}-{stations['coverage'].max():.0%})")

    # 4. weather per kept station
    print("Fetching ERA5 weather ...")
    wx = []
    for _, row in stations.iterrows():
        w = weather.fetch_history(row.lat, row.lon, start, end)
        w["sensor_id"] = row.sensor_id
        wx.append(w)
    save(pd.concat(wx, ignore_index=True), "weather")

    # 5. fires: standard-processing archive, then NRT for the recent gap
    print("Fetching NASA FIRMS fire detections ...")
    sp = firms.to_events(firms.fetch_range(start, end, source="VIIRS_SNPP_SP"))
    if sp.empty:
        print("  S-NPP archive empty; trying NOAA-20")
        sp = firms.to_events(firms.fetch_range(start, end, source="VIIRS_NOAA20_SP"))
    last_sp = sp["time"].max() if not sp.empty else pd.Timestamp(start, tz="UTC")
    gap_start = (last_sp + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    nrt = pd.DataFrame(columns=sp.columns)
    if pd.Timestamp(gap_start) <= pd.Timestamp(end):
        try:
            nrt = firms.to_events(firms.fetch_range(gap_start, end, source="VIIRS_SNPP_NRT"))
        except Exception as e:
            print(f"  NRT gap fill failed ({e}); continuing with archive fires only")
    fires = pd.concat([sp, nrt], ignore_index=True).drop_duplicates()
    save(fires, "fires")
    print(f"  {len(fires):,} fire detections")
    print("Done. Next: python -m smogcast train")


def load_all():
    stations = load("stations")
    pm = clean_pm25(load("pm25_raw"))
    pm = pm[pm["sensor_id"].isin(stations["sensor_id"])]
    return stations, pm, load("weather"), load("fires")
