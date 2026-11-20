from pathlib import Path
import json
import subprocess
import sys
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS = PROJECT_ROOT / "outputs"
ANOMALY_DIR = OUTPUTS / "anomaly_results"
GLOBAL_DIR = OUTPUTS / "global_city_monitor"
DIFFUSION_DIR = OUTPUTS / "diffusion"

LIVE_COMPARISON_FILE = ANOMALY_DIR / "live_model_comparison.csv"
GNN_LIVE_FILE = ANOMALY_DIR / "gnn_live_anomalies.csv"
EVENTS_FILE = ANOMALY_DIR / "tracked_weather_events.csv"
HISTORICAL_GNN_FILE = ANOMALY_DIR / "gnn_historical_anomalies.csv"
HISTORICAL_IF_FILE = ANOMALY_DIR / "historical_anomalies.csv"
COMPARISON_TEST_FILE = ANOMALY_DIR / "model_comparison_test.csv"
COMPARISON_SUMMARY_FILE = ANOMALY_DIR / "anomaly_model_evaluation_summary.json"
COMPARISON_LOCATION_FILE = ANOMALY_DIR / "anomaly_model_evaluation_by_location.csv"
GLOBAL_RESULTS_FILE = GLOBAL_DIR / "global_city_anomaly_results.csv"
GLOBAL_MONITOR_SCRIPT = PROJECT_ROOT / "backend" / "global_city_monitor.py"
DIFFUSION_IMAGE = DIFFUSION_DIR / "generated_weather_comparison.png"
DIFFUSION_ARRAY = DIFFUSION_DIR / "generated_weather.npy"

FEATURE_COLUMNS = [
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "pressure_msl_hpa",
    "wind_gusts_10m_kmh",
    "wind_sin",
    "wind_cos",
]
LOCATION_NAMES = [
    "Visakhapatnam", "Kakinada", "Machilipatnam", "Nellore", "Ongole"
]

st.set_page_config(
    page_title="AI Weather Anomaly Tracking",
    page_icon="🌦️",
    layout="wide",
)


# ============================================================
# 3. MODERN DARK THEME
# ============================================================
st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    [data-testid="stAppViewContainer"] {
        background: #0b1120;
        color: #e7eef9;
    }
    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.92);
    }
    [data-testid="stSidebar"] {
        background: #101a2b;
        border-right: 1px solid #26364d;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: #101a2b;
    }
    [data-testid="stMarkdownContainer"],
    [data-testid="stText"],
    label, p, li {
        color: #d8e2f0;
        font-size: 1.04rem;
        line-height: 1.55;
    }
    h1 {
        color: #f4f8ff !important;
        font-size: 2.15rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.035em;
    }
    h2 {
        color: #eaf2ff !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    h3 {
        color: #dceaff !important;
        font-size: 1.3rem !important;
        font-weight: 650 !important;
    }
    [data-testid="stCaptionContainer"] p {
        color: #9fb0c7 !important;
        font-size: 0.94rem !important;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #15243a, #111d30);
        border: 1px solid #2a3d57;
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 112px;
    }
    [data-testid="stMetricLabel"] p {
        color: #aebfd5 !important;
        font-size: 0.96rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #f3f7ff !important;
        font-size: 1.8rem !important;
        font-weight: 750 !important;
    }
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #2a3d57;
        border-radius: 12px;
        overflow: hidden;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stMultiSelect"] div[data-baseweb="select"] {
        background: #142238;
        border-color: #38516f;
        color: #f1f6ff;
        font-size: 1rem;
    }
    [data-testid="stButton"] button,
    [data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
        border: 1px solid #3b6687;
    }
    [data-testid="stButton"] button[kind="primary"],
    [data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: #087e8b;
        color: white;
        border: 1px solid #1597a4;
    }
    [data-testid="stExpander"] {
        background: #111e31;
        border: 1px solid #293d57;
        border-radius: 12px;
    }
    hr {
        border-color: #2a3b52;
    }
    [data-testid="stRadio"] label p {
        font-size: 1rem !important;
        font-weight: 550;
    }
    [data-testid="stAlert"] {
        border-radius: 12px;
    }
    @media (max-width: 800px) {
        h1 { font-size: 1.75rem !important; }
        h2 { font-size: 1.4rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.45rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. SHARED HELPERS
# ============================================================
def read_csv(path, **kwargs):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return None


def read_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


def show_missing(path):
    st.info(f"Waiting for this output file: `{path.relative_to(PROJECT_ROOT)}`")


def safe_metric(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def show_weather_globe(global_df=None, compact=False):
    """Render a globe using only coordinates and weather results in the city watchlist."""
    watchlist_path = PROJECT_ROOT / "datasets" / "processed" / "global_city_watchlist.json"
    if not watchlist_path.exists():
        st.info("Add cities to the watchlist to display them on the weather globe.")
        return

    try:
        watchlist = json.loads(watchlist_path.read_text(encoding="utf-8"))
        if not watchlist:
            st.info("The city watchlist is empty. Add a city to populate the globe.")
            return

        rows = []
        result_lookup = {}
        if global_df is not None and not global_df.empty:
            for _, result in global_df.iterrows():
                key = (
                    str(result.get("city", "")).strip().casefold(),
                    str(result.get("country", "")).strip().casefold(),
                )
                result_lookup[key] = result

        for city in watchlist:
            latitude = city.get("latitude")
            longitude = city.get("longitude")
            if latitude is None or longitude is None:
                continue
            key = (
                str(city.get("city", "")).strip().casefold(),
                str(city.get("country", "")).strip().casefold(),
            )
            result = result_lookup.get(key)
            row = {
                "City": city.get("city", "Unknown"),
                "Country": city.get("country", ""),
                "Latitude": float(latitude),
                "Longitude": float(longitude),
                "Model status": str(result.get("model_indicated_status", "Not scanned")) if result is not None else "Not scanned",
            }
            if result is not None:
                for source_col, label in [
                    ("temperature_c", "Temperature °C"),
                    ("rainfall_mm", "Rainfall mm"),
                    ("wind_speed_kmh", "Wind speed km/h"),
                    ("weather_time", "Weather time"),
                ]:
                    if source_col in result.index:
                        row[label] = result[source_col]
            rows.append(row)

        if not rows:
            st.info("No valid city coordinates are available for the globe yet.")
            return

        try:
            import plotly.express as px
        except ImportError:
            st.warning("The interactive globe needs Plotly. Install it in your project environment with: pip install plotly")
            return

        globe_df = pd.DataFrame(rows)
        color_map = {
            "High": "#ff5d6c",
            "Elevated": "#ffbd59",
            "Normal": "#27c6b7",
            "Not scanned": "#8ba3c7",
        }
        statuses = globe_df["Model status"].astype(str)
        globe_df["Globe category"] = statuses.map(
            lambda value: "High" if "high" in value.casefold()
            else "Elevated" if "elevated" in value.casefold()
            else "Normal" if "normal" in value.casefold()
            else "Not scanned"
        )
        hover_columns = [column for column in [
            "Temperature °C", "Rainfall mm", "Wind speed km/h", "Weather time"
        ] if column in globe_df.columns]
        fig = px.scatter_geo(
            globe_df,
            lat="Latitude",
            lon="Longitude",
            color="Globe category",
            hover_name="City",
            hover_data={
                "Country": True,
                "Model status": True,
                **{column: True for column in hover_columns},
                "Latitude": False,
                "Longitude": False,
                "Globe category": False,
            },
            color_discrete_map=color_map,
            projection="orthographic",
            title=None,
        )
        fig.update_traces(marker={"size": 10 if compact else 13, "line": {"width": 1, "color": "#e6f1ff"}})
        fig.update_geos(
            showland=True, landcolor="#1b2a41",
            showocean=True, oceancolor="#091629",
            showlakes=True, lakecolor="#10233b",
            showcountries=True, countrycolor="#52647d",
            showcoastlines=True, coastlinecolor="#71839a",
            bgcolor="#0b1120",
            projection_rotation={"lon": 0, "lat": 15, "roll": 0},
        )
        fig.update_layout(
            height=390 if compact else 610,
            margin={"l": 0, "r": 0, "t": 12, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e7eef9"},
            legend={"title": "Model-indicated status", "orientation": "h", "y": 0.02, "x": 0.02},
            geo={"bgcolor": "rgba(0,0,0,0)"},
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Markers show configured watchlist cities only. Hover over a marker for available weather readings. Status colors are model indications, not official warnings.")
    except Exception as exc:
        st.warning(f"The weather globe could not be displayed: {exc}")


def format_time(value):
    try:
        return pd.to_datetime(value).strftime("%d %b %Y, %H:%M")
    except Exception:
        return str(value)


def page_header(title, subtitle):
    st.title(title)
    st.caption(subtitle)


# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🌦️ Weather AI")
st.sidebar.caption("Spatio-temporal anomaly monitoring prototype")
pages = [
    "Dashboard Overview",
    "Live Weather",
    "GNN Anomaly Detection",
    "Model Comparison",
    "Tracked Weather Events",
    "Global City Monitor",
]
selected_page = st.sidebar.radio("Navigate", pages)
st.sidebar.divider()
st.sidebar.caption("Prototype based on historical and live Open-Meteo data.")
st.sidebar.caption("Model-indicated anomalies are not official weather warnings.")

# ============================================================
# 4. DASHBOARD OVERVIEW
# ============================================================
if selected_page == "Dashboard Overview":
    page_header(
        "🌍 AI-Based Weather Anomaly Tracking",
        "Overview of the Andhra Pradesh proof-of-concept and global city watchlist.",
    )
    live = read_csv(LIVE_COMPARISON_FILE)
    events = read_csv(EVENTS_FILE)
    global_df = read_csv(GLOBAL_RESULTS_FILE)

    c1, c2, c3, c4 = st.columns(4)
    if live is not None and not live.empty:
        anomaly_count = int((live.get("combined_status", pd.Series(dtype=str)) != "Normal").sum())
        safe_metric("Reference locations", len(live))
        safe_metric("Live flagged locations", anomaly_count)
        latest = live["time"].iloc[0] if "time" in live.columns else "—"
        safe_metric("Latest reference update", format_time(latest))
    else:
        safe_metric("Reference locations", "—")
        safe_metric("Live flagged locations", "—")
        safe_metric("Latest reference update", "—")
    safe_metric("Tracked candidate groups", len(events) if events is not None else "—")
    st.subheader("🌍 Global weather globe")
    show_weather_globe(global_df, compact=True)

    if global_df is not None and not global_df.empty:
        high = int((global_df["risk_rank"] == 2).sum()) if "risk_rank" in global_df else 0
        elevated = int((global_df["risk_rank"] == 1).sum()) if "risk_rank" in global_df else 0
        st.subheader("Global watchlist snapshot")
        g1, g2, g3 = st.columns(3)
        g1.metric("Cities scanned", len(global_df))
        g2.metric("High model-indicated", high)
        g3.metric("Elevated model-indicated", elevated)
        cols = [c for c in ["monitoring_rank", "city", "country", "model_indicated_status", "if_status", "gnn_status", "weather_time"] if c in global_df.columns]
        st.dataframe(global_df[cols], width="stretch", hide_index=True)
    else:
        st.info("Global city results are not available yet. Open Global City Monitor to run a scan.")

    st.subheader("Project scope and interpretation")
    st.markdown(
        """
        - Historical model development uses five coastal Andhra Pradesh locations (2018–2020).
        - The GNN is a spatial graph autoencoder. It flags unusual reconstruction patterns; it does not forecast future weather.
        - The event tracker groups nearby-in-time detections into **candidate groups**, not confirmed meteorological events.
        - The global city monitor is a finite configurable watchlist, not continuous monitoring of every city worldwide.
        - Global GNN transfer inference is experimental because the model was trained on the five Andhra Pradesh reference locations.
        - “High” and “Elevated” describe model-indicated anomaly status, not official alerts or disaster predictions.
        """
    )

# ============================================================
# 5. LIVE WEATHER
# ============================================================
elif selected_page == "Live Weather":
    page_header("🌤️ Live Weather", "Latest API-derived weather features and reference-location anomaly comparison.")
    live = read_csv(LIVE_COMPARISON_FILE)
    if live is None:
        show_missing(LIVE_COMPARISON_FILE)
    elif live.empty:
        st.warning("The live comparison file is empty.")
    else:
        if "time" in live.columns:
            st.caption(f"Weather timestamp: {format_time(live['time'].iloc[0])}")
        status_col = "combined_status" if "combined_status" in live.columns else None
        if status_col:
            flagged = live[live[status_col] != "Normal"]
            st.metric("Locations with at least one model flag", len(flagged))
            if not flagged.empty:
                st.warning("One or more models flagged unusual conditions at these reference locations.")
                st.dataframe(flagged, width="stretch", hide_index=True)
            else:
                st.success("Neither model currently flags an anomaly at the five reference locations.")
        st.subheader("Live model results")
        st.dataframe(live, width="stretch", hide_index=True)
        st.caption("Open-Meteo current conditions are model-based estimates and are not necessarily direct station observations.")

# ============================================================
# 6. GNN ANOMALY DETECTION
# ============================================================
elif selected_page == "GNN Anomaly Detection":
    page_header("🧠 GNN Anomaly Detection", "Spatial graph autoencoder reconstruction-error results.")
    live_gnn = read_csv(GNN_LIVE_FILE)
    if live_gnn is None:
        show_missing(GNN_LIVE_FILE)
    else:
        if "gnn_anomaly" in live_gnn.columns:
            flagged = live_gnn[live_gnn["gnn_anomaly"].astype(int) == 1]
        else:
            flagged = pd.DataFrame()
        a, b, c = st.columns(3)
        a.metric("Locations evaluated", len(live_gnn))
        b.metric("GNN flags", len(flagged))
        c.metric("Normal", len(live_gnn) - len(flagged))
        st.subheader("Live reconstruction-error scores")
        st.dataframe(live_gnn, width="stretch", hide_index=True)
        if not flagged.empty:
            st.warning("GNN model-indicated anomalies")
            st.dataframe(flagged, width="stretch", hide_index=True)
        else:
            st.success("No GNN anomaly flags in the latest reference-location output.")

    st.divider()
    st.subheader("Historical GNN detections")
    hist = read_csv(HISTORICAL_GNN_FILE)
    if hist is None:
        show_missing(HISTORICAL_GNN_FILE)
    else:
        if "gnn_anomaly" in hist.columns:
            hist_flags = hist[hist["gnn_anomaly"].astype(int) == 1]
            st.metric("Historical flagged records", len(hist_flags))
            if "location_name" in hist_flags.columns:
                by_loc = hist_flags["location_name"].value_counts().rename_axis("Location").reset_index(name="Flagged records")
                st.bar_chart(by_loc.set_index("Location"))
            st.dataframe(hist_flags.head(500), width="stretch", hide_index=True)
            st.caption("Showing at most 500 flagged rows to keep the dashboard responsive.")
        else:
            st.warning("The historical GNN file does not contain a `gnn_anomaly` column.")

# ============================================================
# 7. MODEL COMPARISON
# ============================================================
elif selected_page == "Model Comparison":
    page_header("⚖️ Model Comparison", "Comparison of Isolation Forest and GNN detection behavior.")
    summary = read_json(COMPARISON_SUMMARY_FILE)
    location = read_csv(COMPARISON_LOCATION_FILE)
    test = read_csv(COMPARISON_TEST_FILE)

    if summary is not None:
        st.subheader("Evaluation summary")
        st.json(summary)
    elif test is not None:
        st.caption("Summary JSON not found; displaying the available test comparison data.")
    else:
        show_missing(COMPARISON_TEST_FILE)

    if location is not None:
        st.subheader("Detection rates by location")
        st.dataframe(location, width="stretch", hide_index=True)
        numeric_cols = [c for c in location.columns if c.endswith("_rate") or c.endswith("_pct")]
        if numeric_cols:
            st.bar_chart(location.set_index(location.columns[0])[numeric_cols])

    if test is not None:
        st.subheader("Test-set comparison records")
        st.caption(f"{len(test):,} records available. Showing first 1,000.")
        st.dataframe(test.head(1000), width="stretch", hide_index=True)

    st.warning(
        "There are no ground-truth meteorological anomaly labels in this project. "
        "Agreement, detection counts, and score distributions describe model behavior; "
        "they are not accuracy, precision, recall, F1, or validated weather-detection skill."
    )

# ============================================================
# 8. TRACKED WEATHER EVENTS
# ============================================================
elif selected_page == "Tracked Weather Events":
    page_header("🧭 Tracked Weather Events", "Time-and-neighborhood grouping of historical GNN anomaly detections.")
    events = read_csv(EVENTS_FILE)
    if events is None:
        show_missing(EVENTS_FILE)
    elif events.empty:
        st.info("No tracked event groups are present.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Candidate groups", len(events))
        e2.metric("Multi-record groups", int((events["event_category"] == "Multi-record candidate event").sum()) if "event_category" in events else "—")
        e3.metric("Isolated detections", int((events["event_category"] == "Isolated detection").sum()) if "event_category" in events else "—")

        filtered = events.copy()
        filter_cols = st.columns(3)
        if "severity" in filtered.columns:
            severity_values = sorted(filtered["severity"].dropna().unique().tolist())
            selected_severity = filter_cols[0].multiselect("Severity", severity_values, default=severity_values)
            filtered = filtered[filtered["severity"].isin(selected_severity)]
        if "event_category" in filtered.columns:
            category_values = sorted(filtered["event_category"].dropna().unique().tolist())
            selected_category = filter_cols[1].multiselect("Group type", category_values, default=category_values)
            filtered = filtered[filtered["event_category"].isin(selected_category)]
        if "dominant_weather_signal" in filtered.columns:
            signal_values = sorted(filtered["dominant_weather_signal"].dropna().unique().tolist())
            selected_signal = filter_cols[2].multiselect("Dominant signal", signal_values, default=signal_values)
            filtered = filtered[filtered["dominant_weather_signal"].isin(selected_signal)]

        st.subheader(f"Candidate groups ({len(filtered):,})")
        display_cols = [c for c in [
            "event_id", "start_time", "end_time", "duration_hours",
            "anomaly_records", "number_of_locations", "locations",
            "maximum_gnn_anomaly_score", "severity",
            "dominant_weather_signal", "event_category", "event_status"
        ] if c in filtered.columns]
        st.dataframe(filtered[display_cols], width="stretch", hide_index=True)

        if not filtered.empty and "event_id" in filtered.columns:
            chosen = st.selectbox("Inspect a candidate group", filtered["event_id"].astype(str).tolist())
            row = filtered[filtered["event_id"].astype(str) == chosen].iloc[0]
            st.subheader(f"Inspection: {chosen}")
            details = st.columns(3)
            for col, label, key in zip(
                details,
                ["Start", "End", "Maximum GNN score"],
                ["start_time", "end_time", "maximum_gnn_anomaly_score"],
            ):
                col.metric(label, str(row[key]) if key in row else "—")
            st.write(f"**Locations:** {row.get('locations', '—')}")
            st.write(f"**Severity:** {row.get('severity', '—')}")
            st.write(f"**Dominant signal:** {row.get('dominant_weather_signal', '—')}")
            st.write(f"**Category:** {row.get('event_category', '—')}")
        st.caption(
            "These are algorithmically grouped candidate detections, not confirmed storms or "
            "meteorological events. Severity bins are prototype thresholds."
        )

# ============================================================
# 9. GLOBAL CITY MONITOR
# ============================================================
elif selected_page == "Global City Monitor":
    page_header(
        "🌐 Global City Monitor",
        "Live weather retrieval and model-based anomaly screening for a configurable city watchlist.",
    )
    st.warning(
        "Experimental screening only: the GNN was trained on five Andhra Pradesh locations. "
        "Its six-node transfer inference for other cities is out-of-distribution and is not validated "
        "for global weather risk. These statuses are not official warnings."
    )

    backend_folder = PROJECT_ROOT / "backend"
    watchlist_file = PROJECT_ROOT / "datasets" / "processed" / "global_city_watchlist.json"

    # Load the tested city geocoder from the project's backend directory.
    backend_text = str(backend_folder)
    if backend_text not in sys.path:
        sys.path.insert(0, backend_text)
    try:
        from city_weather import search_city
    except Exception as exc:
        search_city = None
        st.error(f"Could not load the city-search utility: {exc}")

    st.subheader("🔎 Search and add a city")
    with st.form("global_city_search_form", clear_on_submit=False):
        city_query = st.text_input(
            "City name", placeholder="Try Toronto, Nairobi, Singapore..."
        )
        search_submitted = st.form_submit_button("Search cities")

    if search_submitted:
        if not city_query.strip():
            st.warning("Enter at least two characters for a city search.")
        elif search_city is None:
            st.error("City search is unavailable because the backend utility did not load.")
        else:
            try:
                matches = search_city(city_query.strip(), result_count=10)
                st.session_state["global_city_search_matches"] = matches
                if not matches:
                    st.info("No matching cities were found. Try a different spelling.")
            except Exception as exc:
                st.error(f"City search failed: {exc}")

    matches = st.session_state.get("global_city_search_matches", [])
    if matches:
        def city_label(city):
            parts = [city.get("name", "Unknown"), city.get("admin1", ""), city.get("country", "")]
            return ", ".join(part for part in parts if part)

        selected_city_label = st.selectbox(
            "Choose the correct city and region", options=[city_label(x) for x in matches],
            key="global_city_match_select"
        )
        selected_city = matches[[city_label(x) for x in matches].index(selected_city_label)]
        if st.button("➕ Add selected city to watchlist", type="primary"):
            try:
                watchlist_file.parent.mkdir(parents=True, exist_ok=True)
                if watchlist_file.exists():
                    watchlist = json.loads(watchlist_file.read_text(encoding="utf-8"))
                else:
                    watchlist = []
                duplicate = any(
                    item.get("city", "").casefold() == selected_city["name"].casefold()
                    and item.get("country", "").casefold() == selected_city["country"].casefold()
                    for item in watchlist
                )
                if duplicate:
                    st.warning("That city is already in the watchlist.")
                else:
                    watchlist.append({
                        "city": selected_city["name"],
                        "country": selected_city["country"],
                        "admin1": selected_city.get("admin1", ""),
                        "latitude": float(selected_city["latitude"]),
                        "longitude": float(selected_city["longitude"]),
                        "timezone": selected_city.get("timezone", "auto"),
                    })
                    watchlist_file.write_text(
                        json.dumps(watchlist, indent=4, ensure_ascii=False), encoding="utf-8"
                    )
                    st.session_state.pop("global_city_search_matches", None)
                    st.session_state["global_city_force_scan"] = True
                    st.success(f"Added {selected_city['name']}, {selected_city['country']}.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Could not update the watchlist: {exc}")

    # Watchlist removal is explicit and only changes the selected entry.
    if watchlist_file.exists():
        try:
            current_watchlist = json.loads(watchlist_file.read_text(encoding="utf-8"))
            if current_watchlist:
                st.subheader("Current watchlist")
                watchlist_labels = [
                    f"{x.get('city', 'Unknown')}, {x.get('country', '')}" for x in current_watchlist
                ]
                remove_label = st.selectbox(
                    "Select a city to remove", options=["— Keep all cities —"] + watchlist_labels,
                    key="global_city_remove_select"
                )
                if remove_label != "— Keep all cities —" and st.button("Remove selected city"):
                    index = watchlist_labels.index(remove_label)
                    removed = current_watchlist.pop(index)
                    watchlist_file.write_text(
                        json.dumps(current_watchlist, indent=4, ensure_ascii=False), encoding="utf-8"
                    )
                    st.session_state["global_city_force_scan"] = True
                    st.success(f"Removed {removed.get('city')} from the watchlist.")
                    st.rerun()
        except Exception as exc:
            st.error(f"Could not display or update the watchlist: {exc}")

    st.divider()
    st.subheader("⚙️ Monitoring and refresh settings")
    refresh_minutes = st.selectbox(
        "Automatic scan interval", options=[5, 10, 15, 30], index=1,
        format_func=lambda value: f"Every {value} minutes", key="global_city_refresh_interval"
    )
    st.caption(
        "Automatic scans repeat while this dashboard page remains open. Each scan calls the weather API "
        "for the configured watchlist and reference locations. The interval is configurable to limit API traffic."
    )

    def run_global_city_scan():
        if not GLOBAL_MONITOR_SCRIPT.exists():
            raise FileNotFoundError(f"Monitor script not found: {GLOBAL_MONITOR_SCRIPT}")
        completed = subprocess.run(
            [sys.executable, str(GLOBAL_MONITOR_SCRIPT)], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=600,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(details or "Global city scan failed.")
        return completed.stdout

    @st.fragment(run_every=refresh_minutes * 60)
    def global_city_results_fragment():
        manual_scan = st.button("🔄 Scan cities now", type="primary", key="global_city_manual_scan")
        force_scan = st.session_state.pop("global_city_force_scan", False)
        file_is_missing = not GLOBAL_RESULTS_FILE.exists()
        file_is_stale = (
            GLOBAL_RESULTS_FILE.exists()
            and (datetime.now().timestamp() - GLOBAL_RESULTS_FILE.stat().st_mtime) >= refresh_minutes * 60
        )
        if manual_scan or force_scan or file_is_missing or file_is_stale:
            with st.spinner("Fetching current weather and running the anomaly models..."):
                try:
                    scan_log = run_global_city_scan()
                    st.session_state["global_city_last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["global_city_scan_log"] = scan_log
                    st.success("Global city scan completed.")
                except Exception as exc:
                    st.error(f"Global city scan could not be completed: {exc}")

        global_df = read_csv(GLOBAL_RESULTS_FILE)
        if global_df is None:
            show_missing(GLOBAL_RESULTS_FILE)
            return
        if global_df.empty:
            st.info("The global city results file is empty. Run a scan to populate it.")
            return

        st.subheader("🌐 Interactive global weather globe")
        show_weather_globe(global_df, compact=False)

        if "risk_rank" in global_df.columns:
            high_count = int((global_df["risk_rank"] == 2).sum())
            elevated_count = int((global_df["risk_rank"] == 1).sum())
        else:
            high_count = elevated_count = 0
        m1, m2, m3 = st.columns(3)
        m1.metric("Cities in latest scan", len(global_df))
        m2.metric("High model-indicated", high_count)
        m3.metric("Elevated model-indicated", elevated_count)
        if "scan_time" in global_df.columns:
            st.caption(f"Last saved scan: {global_df['scan_time'].iloc[0]}")

        status_col = "model_indicated_status"
        if status_col in global_df.columns:
            high = global_df[global_df[status_col].astype(str).str.contains("High", case=False, na=False)]
            elevated = global_df[global_df[status_col].astype(str).str.contains("Elevated", case=False, na=False)]
            if not high.empty:
                st.error("High model-indicated anomaly status — inspect these cities; this is not an official alert.")
                st.dataframe(high, width="stretch", hide_index=True)
            if not elevated.empty:
                st.warning("Elevated model-indicated status")
                st.dataframe(elevated, width="stretch", hide_index=True)

        st.subheader("Ranked city watchlist")
        st.dataframe(global_df, width="stretch", hide_index=True)
        with st.expander("View latest scan log"):
            log_text = st.session_state.get("global_city_scan_log")
            st.code(log_text if log_text else "No scan log in this session yet.", language="text")

    global_city_results_fragment()
    st.caption(
        "Coverage is the finite configured watchlist, not every city worldwide. Weather values are Open-Meteo "
        "model estimates. Global GNN transfer inference is experimental and out-of-distribution; anomaly flags "
        "are not official hazard alerts. Automatic scans run only while this dashboard page is open."
    )

# ============================================================
# 10. FOOTER
# ============================================================
st.sidebar.divider()
st.sidebar.caption("Academic proof-of-concept • Not an operational forecasting or warning system")
