"""India National AQI (CPCB) for PM2.5, based on 24-hour average concentration."""
from __future__ import annotations

import numpy as np

# (conc_lo, conc_hi, aqi_lo, aqi_hi, category)
BREAKPOINTS = [
    (0, 30, 0, 50, "Good"),
    (30, 60, 50, 100, "Satisfactory"),
    (60, 90, 100, 200, "Moderate"),
    (90, 120, 200, 300, "Poor"),
    (120, 250, 300, 400, "Very Poor"),
    (250, 500, 400, 500, "Severe"),
]
CATEGORIES = [b[4] for b in BREAKPOINTS]
COLORS = {"Good": "#2e9e4f", "Satisfactory": "#8cc63f", "Moderate": "#f2c200",
          "Poor": "#f08a24", "Very Poor": "#d7263d", "Severe": "#7e1e3c"}

ADVICE = {
    "Good": "Air is clean. No restrictions on outdoor activity.",
    "Satisfactory": "Fine for most people; very sensitive people may notice mild discomfort.",
    "Moderate": "People with asthma, heart or lung disease, children and the elderly should cut down long outdoor exertion.",
    "Poor": "Most people may feel breathing discomfort on long exposure. Sensitive groups should avoid outdoor exercise.",
    "Very Poor": "Breathing illness likely on prolonged exposure. Keep outdoor time short; use an N95 mask outdoors.",
    "Severe": "Affects healthy people too and seriously affects those with existing illness. Stay indoors, keep windows shut, avoid all outdoor exertion.",
}


def category(pm25_24h: float) -> str:
    if pm25_24h is None or np.isnan(pm25_24h):
        return "Unknown"
    for lo, hi, _, _, cat in BREAKPOINTS:
        if pm25_24h <= hi:
            return cat
    return "Severe"


def sub_index(pm25_24h: float) -> float:
    if pm25_24h is None or np.isnan(pm25_24h):
        return float("nan")
    for lo, hi, alo, ahi, _ in BREAKPOINTS:
        if pm25_24h <= hi:
            return alo + (ahi - alo) * (pm25_24h - lo) / (hi - lo)
    return 500.0
