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
    if global_df is not None and not global_df.empty:
        high = int((global_df["risk_rank"] == 2).sum()) if "risk_rank" in global_df else 0
        elevated = int((global_df["risk_rank"] == 1).sum()) if "risk_rank" in global_df else 0
        st.subheader("Global watchlist snapshot")
        g1, g2, g3 = st.columns(3)
        g1.metric("Cities scanned", len(global_df))
        g2.metric("High model-indicated", high)
        g3.metric("Elevated model-indicated", elevated)
        cols = [c for c in ["monitoring_rank", "city", "country", "model_indicated_status", "if_status", "gnn_status", "weather_time"] if c in global_df.columns]
        st.dataframe(global_df[cols], use_container_width=True, hide_index=True)
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
                st.dataframe(flagged, use_container_width=True, hide_index=True)
            else:
                st.success("Neither model currently flags an anomaly at the five reference locations.")
        st.subheader("Live model results")
        st.dataframe(live, use_container_width=True, hide_index=True)
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
        st.dataframe(live_gnn, use_container_width=True, hide_index=True)
        if not flagged.empty:
            st.warning("GNN model-indicated anomalies")
            st.dataframe(flagged, use_container_width=True, hide_index=True)
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
            st.dataframe(hist_flags.head(500), use_container_width=True, hide_index=True)
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
        st.dataframe(location, use_container_width=True, hide_index=True)
        numeric_cols = [c for c in location.columns if c.endswith("_rate") or c.endswith("_pct")]
        if numeric_cols:
            st.bar_chart(location.set_index(location.columns[0])[numeric_cols])

    if test is not None:
        st.subheader("Test-set comparison records")
        st.caption(f"{len(test):,} records available. Showing first 1,000.")
        st.dataframe(test.head(1000), use_container_width=True, hide_index=True)

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
        st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

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
    refresh_col, interval_col = st.columns([1, 2])
    with refresh_col:
        run_scan = st.button("🔄 Run city scan now", type="primary")
    with interval_col:
        st.caption("The results below are the most recently saved scan. Automatic refresh is not enabled in this version.")

    if run_scan:
        if not GLOBAL_MONITOR_SCRIPT.exists():
            st.error(f"Monitor script not found: {GLOBAL_MONITOR_SCRIPT}")
        else:
            with st.spinner("Scanning the configured city watchlist. This may take a little while..."):
                try:
                    result = subprocess.run(
                        [sys.executable, str(GLOBAL_MONITOR_SCRIPT)],
                        cwd=str(PROJECT_ROOT),
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    if result.returncode == 0:
                        st.success("City scan completed.")
                        if result.stdout.strip():
                            st.code(result.stdout[-5000:])
                    else:
                        st.error("City scan returned an error.")
                        st.code((result.stderr or result.stdout)[-7000:])
                except subprocess.TimeoutExpired:
                    st.error("The scan exceeded the 10-minute limit.")
                except Exception as exc:
                    st.error(f"Could not run the city monitor: {exc}")

    global_df = read_csv(GLOBAL_RESULTS_FILE)
    if global_df is None:
        show_missing(GLOBAL_RESULTS_FILE)
    elif global_df.empty:
        st.info("The global city results file is empty. Run a scan to populate it.")
    else:
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
            st.caption(f"Last scan: {global_df['scan_time'].iloc[0]}")

        status_col = "model_indicated_status"
        if status_col in global_df.columns:
            high = global_df[global_df[status_col].astype(str).str.contains("High", case=False, na=False)]
            if not high.empty:
                st.error("High model-indicated anomaly status — inspect these cities; this is not an official alert.")
                st.dataframe(high, use_container_width=True, hide_index=True)
            elevated = global_df[global_df[status_col].astype(str).str.contains("Elevated", case=False, na=False)]
            if not elevated.empty:
                st.warning("Elevated model-indicated status")
                st.dataframe(elevated, use_container_width=True, hide_index=True)

        st.subheader("Ranked city watchlist")
        st.dataframe(global_df, use_container_width=True, hide_index=True)

        st.subheader("Search and add a city to the watchlist")
        st.caption("City search and watchlist editing are handled by the backend monitor utilities. Run the monitor after adding a city using the project's watchlist workflow.")
        st.caption("This dashboard displays the configured finite watchlist; it does not monitor every city worldwide.")

# ============================================================
# 10. FOOTER
# ============================================================
st.sidebar.divider()
st.sidebar.caption("Academic proof-of-concept • Not an operational forecasting or warning system")
