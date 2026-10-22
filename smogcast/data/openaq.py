"""Ground-truth PM2.5 from OpenAQ (which mirrors CPCB/DPCC/HSPCB monitors).

Free API key: https://explore.openaq.org/register  ->  put it in .env as OPENAQ_API_KEY
"""
from __future__ import annotations

import pandas as pd

from .. import config
from .http import get

BASE = "https://api.openaq.org/v3"


def _headers() -> dict:
    if not config.OPENAQ_API_KEY:
        raise SystemExit("OPENAQ_API_KEY is missing. Add it to .env (see README).")
    return {"X-API-Key": config.OPENAQ_API_KEY}


def find_stations(bbox=config.NCR_BBOX) -> pd.DataFrame:
    """All monitoring locations in the bbox that have a PM2.5 sensor."""
    rows, page = [], 1
    while True:
        r = get(f"{BASE}/locations", headers=_headers(), params={
            "bbox": ",".join(map(str, bbox)), "limit": 1000, "page": page})
        results = r.json().get("results", [])
        for loc in results:
            for s in loc.get("sensors", []):
                if (s.get("parameter") or {}).get("name") == "pm25":
                    rows.append({
                        "location_id": loc["id"],
                        "station": loc.get("name"),
                        "provider": (loc.get("provider") or {}).get("name"),
                        "lat": loc["coordinates"]["latitude"],
                        "lon": loc["coordinates"]["longitude"],
                        "sensor_id": s["id"],
                        "is_monitor": loc.get("isMonitor", True),
                        "last_seen": ((loc.get("datetimeLast") or {}).get("utc")),
                        "first_seen": ((loc.get("datetimeFirst") or {}).get("utc")),
                    })
        if len(results) < 1000:
            break
        page += 1
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Reference-grade monitors only (skip low-cost sensors), still reporting,
    # and running since before the history window we train on.
    df = df[df["is_monitor"].fillna(True).astype(bool)]
    last = pd.to_datetime(df["last_seen"], utc=True, errors="coerce")
    first = pd.to_datetime(df["first_seen"], utc=True, errors="coerce")
    recent = last > pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=14)
    old_enough = first < pd.Timestamp(config.HISTORY_START, tz="UTC") + pd.Timedelta(days=365)
    df = df[recent.fillna(False) & old_enough.fillna(True)]
    return df.drop_duplicates("sensor_id").reset_index(drop=True)


def sensor_info(sensor_id: int) -> dict:
    """First/last measurement times for a sensor (one small request)."""
    r = get(f"{BASE}/sensors/{sensor_id}", headers=_headers())
    res = (r.json().get("results") or [{}])[0]
    return {"first": (res.get("datetimeFirst") or {}).get("utc"),
            "last": (res.get("datetimeLast") or {}).get("utc")}


def fetch_sensor_hours(sensor_id: int, start: str, end: str) -> pd.DataFrame:
    """Hourly PM2.5 for one sensor between start and end (UTC, ISO dates)."""
    out, page = [], 1
    while True:
        r = get(f"{BASE}/sensors/{sensor_id}/hours", headers=_headers(), params={
            "datetime_from": start, "datetime_to": end, "limit": 1000, "page": page})
        results = r.json().get("results", [])
        for m in results:
            period = m.get("period") or {}
            start_t = period.get("datetimeFrom") or period.get("datetime_from") or {}
            if not start_t.get("utc"):
                continue
            out.append({"time": start_t["utc"], "pm25": m.get("value")})
        if len(results) < 1000:
            break
        page += 1
    df = pd.DataFrame(out, columns=["time", "pm25"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["sensor_id"] = sensor_id
    return df
