"""Project-wide settings. Edit here rather than scattering constants around."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"          # forecasts + metrics read by the dashboard

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    """Tiny .env loader so we don't need python-dotenv."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY", "")
FIRMS_MAP_KEY = os.getenv("FIRMS_MAP_KEY", "")

# --- Study region -----------------------------------------------------------
# Delhi NCR monitoring stations (bbox order: min_lon, min_lat, max_lon, max_lat)
NCR_BBOX = (76.80, 28.35, 77.55, 28.90)
MAX_STATIONS = 15            # keep the stations with the best data coverage
MIN_COVERAGE = 0.35          # fraction of hours with a valid reading (history has a 2022-25 gap)

# Upwind source region for crop-residue fires: Punjab + Haryana (+ a margin)
FIRE_BBOX = (73.8, 28.4, 77.6, 32.6)

# --- History window -----------------------------------------------------------
HISTORY_START = "2019-01-01"     # OpenAQ CPCB data: 2016-Oct 2022 and Feb 2025 onward

# --- Modelling ---------------------------------------------------------------
HORIZONS = [24, 48, 72]                 # hours ahead
QUANTILES = [0.1, 0.9]                  # prediction interval
TEST_START = "2025-10-01"               # hold out the most recent winter season
PM25_MAX_VALID = 999                    # CPCB monitors saturate here

# Fire feature settings
N_SECTORS = 16                          # compass sectors (22.5 degrees each)
FIRE_DIST_BANDS_KM = [(0, 150), (150, 400)]
FIRE_LOOKBACK_DAYS = 2
