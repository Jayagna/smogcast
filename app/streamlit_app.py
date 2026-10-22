"""SmogCast dashboard. Reads the files written by `python -m smogcast forecast`
and `python -m smogcast train`; it never calls an API itself, so it stays fast
and free to host on Streamlit Community Cloud.

Run locally:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from smogcast import aqi  # noqa: E402

OUT = ROOT / "outputs"
IST = "Asia/Kolkata"

st.set_page_config(page_title="SmogCast · Delhi NCR air quality forecast", page_icon="🌫️", layout="wide")


@st.cache_data(ttl=600)
def load_json(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(ttl=600)
def load_csv(name: str):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


latest = load_json("latest.json")
metrics = load_json("metrics.json")
fire_season = load_csv("fire_contribution.csv")
live = load_json("live_skill.json")

st.title("🌫️ SmogCast")
st.caption("72-hour PM2.5 forecasts for Delhi NCR monitoring stations, with an estimate of how much "
           "comes from crop-residue fires in Punjab and Haryana.")

if latest is None:
    st.warning("No forecast yet. Run `python -m smogcast forecast` (or wait for the daily GitHub Action).")
    st.stop()
if latest.get("data") == "synthetic":
    st.error("Demo mode: these numbers come from SYNTHETIC data and are not real forecasts.")

issued = pd.Timestamp(latest["issued_at"]).tz_convert(IST)
generated = pd.Timestamp(latest["generated_at"]).tz_convert(IST)
st.markdown(f"**Updated:** {generated:%d %b %Y, %H:%M} IST · latest station readings: {issued:%d %b, %H:%M} IST")
if latest.get("data_lag_hours", 0) > 12:
    st.info(f"Station data reaches OpenAQ with a delay; the newest readings are "
            f"{latest['data_lag_hours']} h old, so the model forecasts from there "
            f"(up to {latest['data_lag_hours'] + 72} h ahead of the last reading).")

tab_fc, tab_map, tab_model, tab_about = st.tabs(["Forecast", "Map", "How good is it?", "Method"])

stations = latest["stations"]
names = [s["name"] for s in stations]

# --------------------------------------------------------------------------- forecast tab
with tab_fc:
    def _next_day_pm(s):
        d = s["daily"]
        return d[1]["pm25"] if len(d) > 1 else (d[0]["pm25"] if d else 0)

    default = max(range(len(stations)), key=lambda i: _next_day_pm(stations[i]))  # worst station first
    name = st.selectbox("Station", names, index=default)
    s = stations[names.index(name)]
    daily = s["daily"]

    cols = st.columns(1 + len(daily[:3]))
    cols[0].metric("Latest reading", f"{s['latest_pm25']:.0f} µg/m³" if s["latest_pm25"] is not None else "–")
    for c, d in zip(cols[1:], daily[:3]):
        label = pd.Timestamp(d["date"]).strftime("%a %d %b")
        c.metric(f"{label} (avg)", f"{d['pm25']:.0f} µg/m³", d["category"], delta_color="off")

    nxt = daily[1] if len(daily) > 1 else daily[0]
    color = aqi.COLORS.get(nxt["category"], "#888")
    st.markdown(
        f"<div style='border-left:6px solid {color};padding:0.6rem 1rem;background:rgba(127,127,127,0.08);"
        f"border-radius:6px'><b>{pd.Timestamp(nxt['date']):%A}: {nxt['category']} (AQI ≈ {nxt['aqi']})</b><br>"
        f"{nxt['advice']}<br><span style='opacity:0.75'>Estimated from crop fires: "
        f"~{nxt['fire_pm25']:.0f} µg/m³ of {nxt['pm25']:.0f} "
        f"({(nxt['fire_pm25'] / max(nxt['pm25'], 1)):.0%})</span></div>",
        unsafe_allow_html=True)

    hist = pd.DataFrame(s["history"]).assign(kind="Observed")
    fc = pd.DataFrame(s["forecast"]).assign(kind="Forecast")
    for df in (hist, fc):
        df["time"] = pd.to_datetime(df["time"]).dt.tz_convert(IST).dt.tz_localize(None)
    band = alt.Chart(fc).mark_area(opacity=0.2, color="#d7263d").encode(
        x=alt.X("time:T", title=None), y=alt.Y("lo:Q", title="PM2.5 (µg/m³)"), y2="hi:Q")
    line = alt.Chart(pd.concat([hist, fc])).mark_line().encode(
        x="time:T", y="pm25:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Observed", "Forecast"], range=["#555", "#d7263d"]),
                        legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.condition(alt.datum.kind == "Forecast", alt.value([5, 3]), alt.value([0])))
    fire = alt.Chart(fc).mark_area(opacity=0.35, color="#f08a24").encode(x="time:T", y="fire:Q")
    rules = alt.Chart(pd.DataFrame({"y": [60, 120, 250]})).mark_rule(strokeDash=[2, 4], opacity=0.4).encode(y="y:Q")
    st.altair_chart((band + fire + line + rules).properties(height=360), use_container_width=True)
    st.caption("Red band: 80% prediction interval. Orange area: model-estimated crop-fire share. "
               "Dotted lines: 60 / 120 / 250 µg/m³ (Satisfactory / Poor / Very Poor limits for the 24-h average).")

# --------------------------------------------------------------------------- map tab
with tab_map:
    pts = []
    for s in stations:
        d = s["daily"][1] if len(s["daily"]) > 1 else (s["daily"][0] if s["daily"] else None)
        if d is None:
            continue
        hexc = aqi.COLORS.get(d["category"], "#888888").lstrip("#")
        pts.append({"name": s["name"], "lat": s["lat"], "lon": s["lon"], "pm25": d["pm25"],
                    "cat": d["category"], "rgb": [int(hexc[i:i + 2], 16) for i in (0, 2, 4)]})
    fires = pd.DataFrame(latest.get("fires_48h", {}).get("points", []))
    layers = []
    if not fires.empty:
        layers.append(pdk.Layer("ScatterplotLayer", fires, get_position="[lon, lat]", get_radius=2500,
                                get_fill_color=[240, 120, 30, 140], pickable=False))
    layers.append(pdk.Layer("ScatterplotLayer", pd.DataFrame(pts), get_position="[lon, lat]",
                            get_radius=3500, get_fill_color="rgb", pickable=True))
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=29.6, longitude=76.3, zoom=6),
                             tooltip={"text": "{name}\nTomorrow: {pm25} µg/m³ ({cat})"}, map_style=None))
    f48 = latest.get("fires_48h", {})
    st.caption(f"Orange dots: {f48.get('count', 0):,} satellite fire detections in Punjab/Haryana in the last 48 h "
               "(NASA FIRMS VIIRS). Station colours: tomorrow's AQI category.")

# --------------------------------------------------------------------------- evaluation tab
with tab_model:
    if metrics is None:
        st.info("Train the model to see evaluation results.")
    else:
        st.subheader("Next-day average PM2.5 on a held-out season")
        st.caption(f"Test period: {metrics['test_period'][0][:10]} → {metrics['test_period'][1][:10]} · "
                   f"{metrics['n_forecasts']:,} station-day forecasts. The model never saw this period during training.")
        rows = []
        for day, v in metrics["daily_mean"].items():
            for model, m in v.items():
                rows.append({"Horizon": day, "Model": model, "MAE (µg/m³)": round(m["MAE"], 1),
                             "AQI category accuracy": f"{m['category_accuracy']:.0%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        cov = metrics["hourly"]["Day 1 (1-24h)"]["SmogCast"].get("interval_coverage")
        if cov is not None:
            st.caption(f"80% prediction interval actually covered {cov:.0%} of hourly observations on day 1.")

        imp = metrics["importance"]["MAE_increase_when_shuffled"]
        st.subheader("What the model relies on")
        st.altair_chart(alt.Chart(pd.DataFrame({"Feature group": list(imp), "MAE increase": list(imp.values())}))
                        .mark_bar().encode(x="MAE increase:Q", y=alt.Y("Feature group:N", sort="-x"))
                        .properties(height=200), use_container_width=True)

        if fire_season is not None and not fire_season.empty:
            st.subheader("Estimated crop-fire contribution through the test season")
            fs = fire_season.copy()
            fs["date"] = pd.to_datetime(fs["date"])
            long = fs.melt("date", ["pred_no_fire", "fire_contribution"], "part", "pm25")
            long["part"] = long["part"].map({"pred_no_fire": "Without fires", "fire_contribution": "From fires"})
            st.altair_chart(alt.Chart(long).mark_area().encode(
                x=alt.X("date:T", title=None), y=alt.Y("pm25:Q", stack=True, title="NCR mean PM2.5 (µg/m³)"),
                color=alt.Color("part:N", scale=alt.Scale(range=["#9aa5b1", "#f08a24"]),
                                legend=alt.Legend(title=None, orient="top"))).properties(height=300),
                use_container_width=True)
            st.caption("Model-based counterfactual (prediction minus prediction with fire features set to zero). "
                       "An estimate of association, not a chemical source apportionment.")
    if live:
        st.subheader("Live track record")
        st.write(f"Mean absolute error of forecasts actually issued ({live['from'][:10]} → {live['to'][:10]}, "
                 f"{live['n']:,} station-hours): " + ", ".join(f"{k}: {v} µg/m³" for k, v in live["MAE_by_day"].items()))

# --------------------------------------------------------------------------- about tab
with tab_about:
    st.markdown("""
**Data**
- PM2.5: CPCB / DPCC / HSPCB reference monitors via the OpenAQ API (hourly)
- Weather: ERA5 reanalysis for training, Open-Meteo forecasts for live runs (wind, boundary-layer height, humidity, rain)
- Fires: NASA FIRMS VIIRS 375 m active-fire detections over Punjab and Haryana

**Model**
One gradient-boosted model forecasts every horizon from 1 to 72 hours, with the horizon as an input,
plus two quantile models for the 80% interval. Features cover recent PM2.5 at the station and across NCR,
weather now and at the target hour, and an *upwind fire score*: fire radiative power in each compass
sector, weighted by how closely that sector lines up with the wind.

**Evaluation**
Trained on data before the test season (with a 3-day gap) and compared against persistence and
climatology baselines on the AQI-relevant 24-hour average.

**Limitations**
Training uses reanalysis weather while live runs use forecast weather, so live accuracy is expected to be
somewhat worse than the test-season numbers; the live track record above measures the real thing.
""")
