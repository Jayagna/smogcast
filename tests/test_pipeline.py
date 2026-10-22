"""Tests run on synthetic data, so they need no API keys and no network."""
import numpy as np
import pandas as pd

from smogcast import aqi, synthetic
from smogcast.clean import clean_pm25
from smogcast.features import build_panel, haversine_bearing, make_rows, sector_alignment


def test_aqi_categories():
    assert aqi.category(25) == "Good"
    assert aqi.category(60) == "Satisfactory"
    assert aqi.category(200) == "Very Poor"
    assert aqi.category(400) == "Severe"
    assert aqi.sub_index(30) == 50


def test_bearing_points_northwest_to_punjab():
    # Ludhiana is north-west of Delhi
    dist, bearing = haversine_bearing(28.61, 77.21, np.array([30.90]), np.array([75.85]))
    assert 270 < bearing[0] < 360
    assert 280 < dist[0] < 300


def test_upwind_weight_peaks_when_wind_from_fire_direction():
    w = sector_alignment(np.array([315.0]))[0]       # wind FROM the north-west
    assert w.argmax() == 14                           # sector centred on 315 deg
    assert w[6] == 0                                  # south-east sector gets no weight


def test_cleaning_removes_saturation_and_flatlines():
    t = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    v = np.r_[np.linspace(50, 80, 10), [999] * 2, [120] * 8, np.linspace(100, 60, 10)]
    out = clean_pm25(pd.DataFrame({"time": t, "sensor_id": 1, "pm25": v}))
    assert not (out["pm25"] >= 999).any()
    assert out.loc[12:19, "pm25_obs"].sum() == 0      # 8-hour flatline flagged


def test_no_future_leakage_in_issue_features():
    st, pm, wx, fires = synthetic.make("2025-01-01", "2025-02-15", n_stations=2)
    pm = clean_pm25(pm)
    panel = build_panel(pm, st, wx, fires)
    rows = make_rows(panel, [24], issue_mask=lambda g: g["time"].dt.hour == 0).dropna(subset=["y"])
    # changing future PM2.5 must not change issue-time features
    pm2 = pm.copy()
    cut = pd.Timestamp("2025-02-01", tz="UTC")
    pm2.loc[pm2["time"] > cut, "pm25"] *= 3
    rows2 = make_rows(build_panel(pm2, st, wx, fires), [24],
                      issue_mask=lambda g: g["time"].dt.hour == 0).dropna(subset=["y"])
    a = rows[rows["time"] <= cut].filter(like="pm_")
    b = rows2[rows2["time"] <= cut].filter(like="pm_")
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
