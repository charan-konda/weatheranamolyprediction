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