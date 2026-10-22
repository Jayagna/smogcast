"""Feature engineering.

The model is a single *multi-horizon* regressor: each training row is
(station, issue time t, horizon h) and the target is log1p(PM2.5) at t+h.
Features only use information that is available at time t, except weather at
t+h, which in live use comes from a weather *forecast*.

Feature groups
--------------
pm_*      PM2.5 history at the station and across NCR (log space)
met_*     weather at issue time; tgt_met_* weather around the target hour
fire_*    crop-fire activity in Punjab/Haryana, incl. an *upwind* score that
          weights each fire by how well it lines up with the wind direction
cal_*     calendar of the target hour (IST)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .clean import wind_uv

IST = "Asia/Kolkata"
EARTH_R = 6371.0
_SECTOR_COLS = [f"_fs_{b}_{s}" for b in range(len(config.FIRE_DIST_BANDS_KM))
                for s in range(config.N_SECTORS)]


# --------------------------------------------------------------------------- geometry
def haversine_bearing(lat0, lon0, lat, lon):
    """Distance (km) and initial bearing (deg, 0=N, clockwise) from (lat0,lon0) to points."""
    p0, p1 = np.deg2rad(lat0), np.deg2rad(lat)
    dl = np.deg2rad(lon - lon0)
    a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dl / 2) ** 2
    dist = 2 * EARTH_R * np.arcsin(np.sqrt(a))
    y = np.sin(dl) * np.cos(p1)
    x = np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dl)
    bearing = (np.rad2deg(np.arctan2(y, x)) + 360) % 360
    return dist, bearing


def sector_alignment(wind_from_deg: np.ndarray) -> np.ndarray:
    """Weight of each compass sector for a given wind direction.
    A fire in sector s is 'upwind' when the wind blows FROM s towards the station.
    Returns array [n_times, n_sectors] with weights in [0, 1]."""
    centers = np.arange(config.N_SECTORS) * (360 / config.N_SECTORS)
    diff = np.deg2rad(wind_from_deg[:, None] - centers[None, :])
    return np.clip(np.cos(diff), 0, None) ** 2


def _vector_mean_dir(speed: pd.Series, direction: pd.Series, window: int, center_shift: int = 0):
    u, v = wind_uv(speed, direction)
    um = u.rolling(window, min_periods=1).mean().shift(center_shift)
    vm = v.rolling(window, min_periods=1).mean().shift(center_shift)
    # direction the wind comes FROM
    return (np.rad2deg(np.arctan2(-um, -vm)) + 360) % 360


# --------------------------------------------------------------------------- fires
def fire_sector_sums(fires: pd.DataFrame, lat: float, lon: float, index: pd.DatetimeIndex) -> np.ndarray:
    """Rolling FRP sums per (distance band, sector) over the lookback window.
    Returns array [len(index), n_bands, n_sectors]."""
    nb, ns = len(config.FIRE_DIST_BANDS_KM), config.N_SECTORS
    out = np.zeros((len(index), nb, ns))
    if fires.empty:
        return out
    dist, bearing = haversine_bearing(lat, lon, fires["lat"].values, fires["lon"].values)
    sector = (np.floor((bearing + 180 / ns) / (360 / ns)) % ns).astype(int)
    t = pd.to_datetime(fires["time"], utc=True).dt.floor("h")
    lookback = config.FIRE_LOOKBACK_DAYS * 24
    for b, (lo, hi) in enumerate(config.FIRE_DIST_BANDS_KM):
        m = (dist >= lo) & (dist < hi)
        if not m.any():
            continue
        df = pd.DataFrame({"time": t[m].values, "sector": sector[m], "frp": fires["frp"].values[m]})
        hourly = df.pivot_table(index="time", columns="sector", values="frp", aggfunc="sum")
        hourly = hourly.reindex(columns=range(ns), fill_value=0)
        hourly.index = pd.DatetimeIndex(hourly.index).tz_convert("UTC") if hourly.index.tz else pd.DatetimeIndex(hourly.index).tz_localize("UTC")
        hourly = hourly.reindex(index, fill_value=0).fillna(0)
        out[:, b, :] = hourly.rolling(lookback, min_periods=1).sum().values
    return out


# --------------------------------------------------------------------------- panel
def build_panel(pm: pd.DataFrame, stations: pd.DataFrame, weather: pd.DataFrame,
                fires: pd.DataFrame, index: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """One row per (sensor_id, hour) with issue-time features plus the raw
    columns needed to build target-time features later."""
    if index is None:
        index = pd.date_range(pm["time"].min(), pm["time"].max(), freq="h", tz="UTC")

    # NCR-wide mean (regional background)
    wide = pm.pivot_table(index="time", columns="sensor_id", values="pm25").reindex(index)
    ncr_log = np.log1p(wide).mean(axis=1)

    frames = []
    for _, st in stations.iterrows():
        sid = st["sensor_id"]
        g = pd.DataFrame(index=index)
        g["sensor_id"] = sid
        s = pm[pm["sensor_id"] == sid].set_index("time")["pm25"].reindex(index)
        ls = np.log1p(s)
        g["pm25"] = s
        g["pm_log_now"] = ls
        for lag in (1, 2, 3, 6, 12, 24, 48, 72, 168):
            g[f"pm_log_lag{lag}"] = ls.shift(lag)
        for w in (6, 24, 72, 168):
            g[f"pm_log_mean{w}"] = ls.rolling(w, min_periods=w // 2).mean()
        g["pm_log_std24"] = ls.rolling(24, min_periods=12).std()
        g["pm_log_delta24"] = ls - ls.shift(24)
        g["pm_ncr_log_now"] = ncr_log
        g["pm_ncr_log_mean24"] = ncr_log.rolling(24, min_periods=12).mean()
        g["pm_station_minus_ncr"] = ls - ncr_log

        w = weather[weather["sensor_id"] == sid].set_index("time").reindex(index)
        w = w.interpolate(limit=6, limit_area="inside")
        u, v = wind_uv(w["wind_speed_10m"], w["wind_direction_10m"])
        g["met_u"], g["met_v"] = u, v
        g["met_ws"] = w["wind_speed_10m"]
        g["met_blh"] = w["boundary_layer_height"]
        g["met_rh"] = w["relative_humidity_2m"]
        g["met_t2m"] = w["temperature_2m"]
        g["met_precip24"] = w["precipitation"].rolling(24, min_periods=1).sum()
        g["met_ws_mean24"] = w["wind_speed_10m"].rolling(24, min_periods=6).mean()
        g["met_blh_mean24"] = w["boundary_layer_height"].rolling(24, min_periods=6).mean()
        g["met_vent_index"] = w["wind_speed_10m"] * w["boundary_layer_height"]  # ventilation coefficient
        # raw columns used for target-time features
        for c in ["wind_speed_10m", "wind_direction_10m", "boundary_layer_height",
                  "relative_humidity_2m", "temperature_2m", "precipitation"]:
            g[f"_raw_{c}"] = w[c]

        # fires
        sums = fire_sector_sums(fires, st["lat"], st["lon"], index)       # [T, B, S]
        g["fire_frp_total"] = np.log1p(sums.sum(axis=(1, 2)))
        dir24 = _vector_mean_dir(w["wind_speed_10m"], w["wind_direction_10m"], 24).fillna(0).values
        align = sector_alignment(dir24)                                      # [T, S]
        for b, (lo, hi) in enumerate(config.FIRE_DIST_BANDS_KM):
            g[f"fire_upwind_{lo}_{hi}km"] = np.log1p((sums[:, b, :] * align).sum(axis=1))
        # keep sector sums (as hidden columns) to compute the target-time upwind score
        flat = sums.reshape(len(index), -1).astype("float32")
        g = pd.concat([g, pd.DataFrame(flat, index=index, columns=_SECTOR_COLS)], axis=1)
        num = g.select_dtypes("float64").columns
        g[num] = g[num].astype("float32")              # ~half the memory on multi-year data
        frames.append(g)

    panel = pd.concat(frames)
    panel.index.name = "time"
    panel = panel.reset_index()
    st = stations[["sensor_id", "lat", "lon"]].copy()
    st["station_code"] = np.arange(len(st))
    return panel.merge(st, on="sensor_id", how="left")


def _calendar(times: pd.Series) -> pd.DataFrame:
    t = times.dt.tz_convert(IST)
    doy = t.dt.dayofyear
    return pd.DataFrame({
        "cal_hour": t.dt.hour.values,
        "cal_dow": t.dt.dayofweek.values,
        "cal_doy_sin": np.sin(2 * np.pi * doy / 365.25).values,
        "cal_doy_cos": np.cos(2 * np.pi * doy / 365.25).values,
        "cal_month": t.dt.month.values,
    }, index=times.index)


def make_rows(panel: pd.DataFrame, horizons, issue_mask=None, with_target: bool = True) -> pd.DataFrame:
    """Expand panel into (issue time, horizon) rows."""
    nb, ns = len(config.FIRE_DIST_BANDS_KM), config.N_SECTORS
    rows = []
    for sid, g in panel.groupby("sensor_id", sort=False):
        g = g.sort_values("time").reset_index(drop=True)
        ws, wd = g["_raw_wind_speed_10m"], g["_raw_wind_direction_10m"]
        sums = g[_SECTOR_COLS].to_numpy().reshape(len(g), nb, ns)
        issue = g if issue_mask is None else g[issue_mask(g)]
        issue = issue[issue["pm_log_now"].notna()]         # no reading now -> no forecast row
        base_cols = [c for c in g.columns if not c.startswith("_") and c != "pm25"]
        for h in horizons:
            tgt_idx = issue.index + h
            valid = tgt_idx < len(g)
            iss = issue[valid]
            ti = tgt_idx[valid]
            r = iss[base_cols].copy()
            r["horizon"] = h
            r["target_time"] = g["time"].values[ti]
            # weather around the target hour (from a forecast at run time)
            r["tgt_met_ws"] = ws.values[ti]
            r["tgt_met_blh"] = g["_raw_boundary_layer_height"].values[ti]
            r["tgt_met_rh"] = g["_raw_relative_humidity_2m"].values[ti]
            r["tgt_met_t2m"] = g["_raw_temperature_2m"].values[ti]
            u, v = wind_uv(ws, wd)
            r["tgt_met_u"], r["tgt_met_v"] = u.values[ti], v.values[ti]
            ws12 = ws.rolling(12, min_periods=1).mean().values
            blh12 = g["_raw_boundary_layer_height"].rolling(12, min_periods=1).mean().values
            r["tgt_met_ws_mean12"], r["tgt_met_blh_mean12"] = ws12[ti], blh12[ti]
            r["tgt_met_precip12"] = g["_raw_precipitation"].rolling(12, min_periods=1).sum().values[ti]
            # upwind fire score using fires seen by t and the wind expected at t+h
            dir_tgt = _vector_mean_dir(ws, wd, 24).fillna(0).values[ti]
            al = sector_alignment(dir_tgt)
            for b, (lo, hi) in enumerate(config.FIRE_DIST_BANDS_KM):
                r[f"fire_tgt_upwind_{lo}_{hi}km"] = np.log1p((sums[iss.index, b, :] * al).sum(axis=1))
            cal = _calendar(pd.Series(pd.to_datetime(r["target_time"], utc=True), index=r.index))
            r = pd.concat([r, cal], axis=1)
            if with_target:
                r["y"] = np.log1p(g["pm25"].values[ti])
            num = r.select_dtypes("float64").columns
            r[num] = r[num].astype("float32")            # halves memory on real data
            rows.append(r)
    out = pd.concat(rows, ignore_index=True)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("pm_", "met_", "tgt_met_", "fire_", "cal_")
    return [c for c in df.columns if c.startswith(prefixes)] + ["horizon", "station_code", "lat", "lon"]


FIRE_PREFIX = "fire_"
