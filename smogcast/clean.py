"""Quality control for raw monitor data.

CPCB-style monitors have well-known failure modes: saturation at 999,
zeros/negatives during calibration, 'stuck' flatlines, and isolated spikes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def clean_pm25(df: pd.DataFrame, max_gap_hours: int = 3) -> pd.DataFrame:
    """df: [time, sensor_id, pm25] -> hourly grid per sensor with bad values removed
    and short gaps interpolated. Adds `pm25_obs` (bool) = value was actually measured."""
    df = df.copy()
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce").astype("float64")  # API can return nulls
    df = df.dropna(subset=["pm25"])
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.floor("h")
    df = df.groupby(["sensor_id", "time"], as_index=False)["pm25"].mean()

    out = []
    for sid, g in df.groupby("sensor_id"):
        s = g.set_index("time")["pm25"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC"))

        bad = (s <= 1) | (s >= config.PM25_MAX_VALID)
        # flatline: same value for 6+ consecutive hours
        run_id = (s.diff() != 0).cumsum()
        run_len = s.groupby(run_id).transform("size")
        bad |= (run_len >= 6) & s.notna()
        # isolated spike: >4x both neighbours and > 300
        prev, nxt = s.shift(1), s.shift(-1)
        bad |= (s > 300) & (s > 4 * prev) & (s > 4 * nxt)
        s = s.mask(bad)

        obs = s.notna()
        s = s.interpolate(limit=max_gap_hours, limit_area="inside")
        out.append(pd.DataFrame({"time": s.index, "sensor_id": sid,
                                 "pm25": s.values, "pm25_obs": obs.values}))
    return pd.concat(out, ignore_index=True)


def coverage(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Fraction of hours in [start, end] with a valid reading, per sensor."""
    n = len(pd.date_range(start, end, freq="h", tz="UTC"))
    m = df[df["pm25_obs"] & df["time"].between(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))]
    return (m.groupby("sensor_id").size() / n).clip(upper=1.0)


def wind_uv(speed: pd.Series, direction_deg: pd.Series):
    """Meteorological direction (where wind comes FROM) -> u, v components."""
    rad = np.deg2rad(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)
