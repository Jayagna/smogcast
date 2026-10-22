"""Synthetic data with the same schema as the real pipeline.

Used ONLY for tests and for checking the code runs end to end without API keys.
Never report numbers from this data as results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.weather import VARS


def make(start="2024-01-01", end="2025-12-31", n_stations=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    T = len(idx)
    doy = idx.dayofyear.values
    hour = idx.hour.values

    stations = pd.DataFrame({
        "location_id": np.arange(n_stations) + 100,
        "station": [f"Synthetic {i}" for i in range(n_stations)],
        "lat": 28.6 + rng.normal(0, 0.08, n_stations),
        "lon": 77.2 + rng.normal(0, 0.08, n_stations),
        "sensor_id": np.arange(n_stations) + 1000,
    })

    # weather shared across stations + noise
    winter = np.cos(2 * np.pi * (doy - 15) / 365.25)          # +1 in Jan
    ws = np.clip(2.5 - 1.0 * winter + rng.gamma(2, 0.6, T), 0.1, None)
    wd = (300 + 40 * rng.standard_normal(T)) % 360           # mostly NW
    blh = np.clip(800 - 400 * winter + 600 * np.sin(np.pi * (hour - 1) / 24) ** 4 + rng.normal(0, 100, T), 50, None)

    weather = []
    for sid in stations["sensor_id"]:
        weather.append(pd.DataFrame({
            "time": idx, "sensor_id": sid,
            "temperature_2m": 25 - 10 * winter + rng.normal(0, 1, T),
            "relative_humidity_2m": np.clip(55 + 15 * winter + rng.normal(0, 8, T), 5, 100),
            "wind_speed_10m": ws, "wind_direction_10m": wd,
            "boundary_layer_height": blh,
            "precipitation": rng.exponential(0.05, T) * (rng.random(T) < 0.05),
            "surface_pressure": 990 + rng.normal(0, 3, T),
        }))
    weather = pd.concat(weather, ignore_index=True)[["time", "sensor_id", *VARS]]

    # fires: Oct-Nov burst in Punjab/Haryana
    days = pd.date_range(start, end, freq="D", tz="UTC")
    fire_rows = []
    for d in days:
        season = np.exp(-((d.dayofyear - 310) / 15) ** 2) + 0.4 * np.exp(-((d.dayofyear - 120) / 12) ** 2)
        n = rng.poisson(400 * season)
        if n:
            fire_rows.append(pd.DataFrame({
                "time": d + pd.Timedelta(hours=8) + pd.to_timedelta(rng.integers(0, 60, n), unit="m"),
                "lat": rng.uniform(29.5, 31.5, n), "lon": rng.uniform(74.5, 76.5, n),
                "frp": rng.gamma(2, 5, n)}))
    fires = (pd.concat(fire_rows, ignore_index=True) if fire_rows
             else pd.DataFrame({"time": pd.to_datetime([], utc=True), "lat": [], "lon": [], "frp": []}))

    # PM2.5: background seasonal + stagnation + fire smoke (lagged) + AR noise
    daily_frp = fires.set_index("time")["frp"].resample("h").sum().reindex(idx, fill_value=0)
    smoke = daily_frp.rolling(48, min_periods=1).sum().values / 50
    nw = np.clip(np.cos(np.deg2rad(wd - 315)), 0, None)
    pm_rows = []
    for i, sid in enumerate(stations["sensor_id"]):
        noise = np.zeros(T)
        e = rng.normal(0, 0.15, T)
        for t in range(1, T):
            noise[t] = 0.92 * noise[t - 1] + e[t]
        base = 75 + 60 * winter + 400 / (ws * blh / 300 + 1) + smoke * nw * 1.5
        pm = base * np.exp(noise) * (1 + 0.1 * i)
        pm[rng.random(T) < 0.03] = np.nan
        pm_rows.append(pd.DataFrame({"time": idx, "sensor_id": sid, "pm25": pm}))
    pm = pd.concat(pm_rows, ignore_index=True)
    return stations, pm, weather, fires
