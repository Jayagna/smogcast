"""Meteorology from Open-Meteo (no API key needed).

History: ERA5 reanalysis via the archive API.
Live:    the forecast API, which also returns the past few days, so one call
         gives both the recent observed-ish weather and the next 3 days.
"""
from __future__ import annotations

import pandas as pd

from .http import get

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"

VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "boundary_layer_height", "precipitation",
    "surface_pressure",
]


def _to_frame(js: dict) -> pd.DataFrame:
    h = js["hourly"]
    df = pd.DataFrame({"time": h["time"]})
    for k in VARS:
        df[k] = h.get(k)                      # missing variable -> NaN column
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


def fetch_history(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Year-by-year so each request stays small."""
    out, s = [], pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s <= e:
        n = min(s + pd.DateOffset(years=1) - pd.Timedelta(days=1), e)
        out.append(_fetch_history_chunk(lat, lon, s.strftime("%Y-%m-%d"), n.strftime("%Y-%m-%d")))
        s = n + pd.Timedelta(days=1)
    return pd.concat(out, ignore_index=True)


def _fetch_history_chunk(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": ",".join(VARS), "timezone": "UTC", "wind_speed_unit": "ms",
              "models": "era5"}
    try:
        r = get(ARCHIVE, min_interval=0.3, params=params)
    except Exception:
        # some variables may be unavailable for a model; fall back to the default blend
        params.pop("models")
        r = get(ARCHIVE, min_interval=0.3, params=params)
    return _to_frame(r.json())


def fetch_forecast(lat: float, lon: float, past_days: int = 7, forecast_days: int = 4) -> pd.DataFrame:
    r = get(FORECAST, min_interval=0.3, params={
        "latitude": lat, "longitude": lon, "past_days": past_days,
        "forecast_days": forecast_days, "hourly": ",".join(VARS),
        "timezone": "UTC", "wind_speed_unit": "ms"})
    return _to_frame(r.json())
