"""Active-fire detections from NASA FIRMS (VIIRS 375 m).

Free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/  ->  .env as FIRMS_MAP_KEY
History uses the standard-processing (SP) archive; the live run uses near-real-time (NRT).
The area API returns at most 5 days per call, so history is fetched in 5-day chunks.
"""
from __future__ import annotations

import io

import pandas as pd

from .. import config
from .http import get

BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
KEEP = ["latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "daynight"]


def _key() -> str:
    if not config.FIRMS_MAP_KEY:
        raise SystemExit("FIRMS_MAP_KEY is missing. Add it to .env (see README).")
    return config.FIRMS_MAP_KEY


def _parse(text: str) -> pd.DataFrame:
    if not text.strip() or text.lstrip().lower().startswith(("invalid", "error")):
        return pd.DataFrame(columns=KEEP)
    df = pd.read_csv(io.StringIO(text))
    return df[[c for c in KEEP if c in df.columns]]


def fetch_range(start: str, end: str, source: str = "VIIRS_SNPP_SP",
                bbox=config.FIRE_BBOX) -> pd.DataFrame:
    """The area API accepts at most 5 days per call. Chunks outside the archive's
    coverage return HTTP 400; those are skipped (and reported) instead of aborting."""
    area = ",".join(map(str, bbox))
    chunks, bad = [], []
    for day in pd.date_range(start, end, freq="5D"):
        n = min(5, (pd.Timestamp(end) - day).days + 1)
        url = f"{BASE}/{_key()}/{source}/{area}/{n}/{day:%Y-%m-%d}"
        try:
            chunks.append(_parse(get(url, min_interval=0.5).text))
        except Exception:
            bad.append(day.strftime("%Y-%m-%d"))
    if bad:
        print(f"    {source}: {len(bad)} of {len(bad) + len(chunks)} 5-day chunks unavailable "
              f"(first {bad[0]}, last {bad[-1]})")
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=KEEP)


def fetch_recent(days: int = 3, source: str = "VIIRS_SNPP_NRT", bbox=config.FIRE_BBOX) -> pd.DataFrame:
    area = ",".join(map(str, bbox))
    return _parse(get(f"{BASE}/{_key()}/{source}/{area}/{days}").text)


def to_events(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to one row per fire with a UTC timestamp. Drops low-confidence pixels."""
    if df.empty:
        return pd.DataFrame(columns=["time", "lat", "lon", "frp"])
    conf = df["confidence"].astype(str).str.lower()
    df = df[~conf.isin(["l", "low"])]
    t = df["acq_time"].astype(int).astype(str).str.zfill(4)
    time = pd.to_datetime(df["acq_date"] + " " + t.str[:2] + ":" + t.str[2:], utc=True)
    return pd.DataFrame({"time": time.values, "lat": df["latitude"].values,
                         "lon": df["longitude"].values,
                         "frp": df["frp"].fillna(0).values}).reset_index(drop=True)
