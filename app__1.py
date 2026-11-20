
from pathlib import Path
import json
import sys
import subprocess
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_LIVE_FILE = (
    PROJECT_ROOT / "datasets" / "raw" / "live_weather.csv"
)

FEATURE_LIVE_FILE = (
    PROJECT_ROOT / "datasets" / "raw" / "live_features.csv"
)

GRAPH_FILE = (
    PROJECT_ROOT / "outputs" / "weather_graph"
    / "weighted_adjacency_matrix.csv"
)

GNN_MODEL_FILE = (
    PROJECT_ROOT / "models" / "gnn_autoencoder.pth"
)

GNN_THRESHOLD_FILE = (
    PROJECT_ROOT / "models" / "gnn_threshold.json"
)

IF_MODEL_FILE = (
    PROJECT_ROOT / "models" / "isolation_forest.pkl"
)

IF_THRESHOLD_FILE = (
    PROJECT_ROOT / "models" / "if_threshold.json"
)

FEATURE_COLUMNS = [
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "pressure_msl_hpa",
    "wind_gusts_10m_kmh",
    "wind_sin",
    "wind_cos"
]

LOCATION_IDS = [0, 1, 2, 3, 4]

LOCATION_NAMES = [
    "Visakhapatnam",
    "Kakinada",
    "Machilipatnam",
    "Nellore",
    "Ongole"
]

DEVICE = torch.device("cpu")


# ==================================================
# 2. STREAMLIT CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Weather Anomaly Tracking",
    page_icon="🌦️",
    layout="wide"
)


# ==================================================
# LIGHT THEME
# ==================================================

st.markdown(
    """
    <style>
    /* Main application surfaces */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background-color: #ffffff !important;
        color: #202124 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: #f4f6f8 !important;
        color: #202124 !important;
    }

    /* General text */
    [data-testid="stAppViewContainer"] *,
    [data-testid="stSidebar"] * {
        color: #202124;
    }

    /* Inputs */
    input, textarea, [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #202124 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #f1f3f4 !important;
        color: #202124 !important;
        border: 1px solid #c7cdd3 !important;
    }

    /* Tables and dataframe areas */
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background-color: #ffffff !important;
    }

    /* Expanders and metric cards */
    [data-testid="stExpander"],
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border-color: #d9dee3 !important;
    }

    /* Keep links visible on the light background */
    a {
        color: #1769aa !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 3. GNN ARCHITECTURE
# ==================================================

class SpatialGraphAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.self_layer_1 = nn.Linear(
            input_dim, hidden_dim
        )

        self.neighbor_layer_1 = nn.Linear(
            input_dim, hidden_dim
        )

        self.self_layer_2 = nn.Linear(
            hidden_dim, hidden_dim
        )

        self.neighbor_layer_2 = nn.Linear(
            hidden_dim, hidden_dim
        )

        self.decoder = nn.Linear(
            hidden_dim, input_dim
        )

        self.activation = nn.ReLU()

    def forward(self, x, adjacency):

        neighbor_features = torch.matmul(
            adjacency, x
        )

        hidden = self.activation(
            self.self_layer_1(x)
            + self.neighbor_layer_1(neighbor_features)
        )

        neighbor_hidden = torch.matmul(
            adjacency, hidden
        )

        hidden = self.activation(
            self.self_layer_2(hidden)
            + self.neighbor_layer_2(neighbor_hidden)
        )

        reconstructed = self.decoder(hidden)

        return reconstructed


# ==================================================
# 4. LOAD RAW WEATHER DATA
# ==================================================

@st.cache_data(ttl=30)
def load_raw_weather():

    if not RAW_LIVE_FILE.exists():
        return None

    data = pd.read_csv(RAW_LIVE_FILE)

    if data.empty:
        return None

    data["time"] = pd.to_datetime(
        data["time"],
        errors="coerce"
    )

    data = data.dropna(subset=["time"])

    latest = (
        data.sort_values("time")
        .groupby("location_name", as_index=False)
        .tail(1)
        .sort_values("location_name")
    )

    return latest


# ==================================================
# 5. LOAD STANDARDIZED LIVE FEATURES
# ==================================================

@st.cache_data(ttl=30)
def load_model_features():

    if not FEATURE_LIVE_FILE.exists():
        return None

    data = pd.read_csv(FEATURE_LIVE_FILE)

    if data.empty:
        return None

    required_columns = (
        ["location_id", "location_name", "time"]
        + FEATURE_COLUMNS
    )

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model feature columns: {missing}"
        )

    data["time"] = pd.to_datetime(
        data["time"],
        errors="coerce"
    )

    data = data.dropna(subset=["time"])

    data = data.sort_values(
        "location_id"
    ).reset_index(drop=True)

    return data


# ==================================================
# 6. LOAD GNN MODEL AND GRAPH
# ==================================================

@st.cache_resource
def load_gnn_model():

    if not GNN_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"GNN model not found: {GNN_MODEL_FILE}"
        )

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph file not found: {GRAPH_FILE}"
        )

    if not GNN_THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"GNN threshold not found: {GNN_THRESHOLD_FILE}"
        )

    checkpoint = torch.load(
        GNN_MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )

    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    saved_features = checkpoint["feature_columns"]

    if saved_features != FEATURE_COLUMNS:
        raise ValueError(
            "Feature order does not match the trained GNN."
        )

    if input_dim != len(FEATURE_COLUMNS):
        raise ValueError(
            "GNN input dimension does not match feature count."
        )

    model = SpatialGraphAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    adjacency_df = pd.read_csv(
        GRAPH_FILE,
        index_col=0
    )

    adjacency_df = adjacency_df.loc[
        LOCATION_NAMES,
        LOCATION_NAMES
    ]

    adjacency = adjacency_df.to_numpy(
        dtype=np.float32
    )

    adjacency = (
        adjacency + np.eye(len(LOCATION_IDS))
    )

    degrees = adjacency.sum(axis=1)

    degree_inverse_sqrt = np.diag(
        1.0 / np.sqrt(degrees)
    )

    normalized_adjacency = (
        degree_inverse_sqrt
        @ adjacency
        @ degree_inverse_sqrt
    )

    adjacency_tensor = torch.tensor(
        normalized_adjacency,
        dtype=torch.float32,
        device=DEVICE
    )

    with open(GNN_THRESHOLD_FILE, "r") as file:
        threshold_data = json.load(file)

    threshold = float(
        threshold_data["threshold"]
    )

    return model, adjacency_tensor, threshold


# ==================================================
# 7. GNN LIVE PREDICTION
# ==================================================

def predict_gnn_anomalies(
    model,
    adjacency,
    live_features,
    threshold
):

    live_features = live_features.sort_values(
        "location_id"
    ).reset_index(drop=True)

    if live_features["location_id"].tolist() != LOCATION_IDS:
        raise ValueError(
            "Live features must contain all five locations "
            "in the expected location ID order."
        )

    if live_features["location_name"].tolist() != LOCATION_NAMES:
        raise ValueError(
            "Live location names do not match training order."
        )

    feature_array = live_features[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    if not np.isfinite(feature_array).all():
        raise ValueError(
            "Live model features contain invalid values."
        )

    input_tensor = torch.tensor(
        feature_array[np.newaxis, :, :],
        dtype=torch.float32,
        device=DEVICE
    )

    model.eval()

    with torch.no_grad():

        reconstructed = model(
            input_tensor,
            adjacency
        )

        errors = torch.mean(
            (input_tensor - reconstructed) ** 2,
            dim=2
        )

    scores = errors.cpu().numpy()[0]

    results = live_features[
        ["location_id", "location_name", "time"]
    ].copy()

    results["gnn_anomaly_score"] = scores

    results["gnn_anomaly"] = (
        results["gnn_anomaly_score"] > threshold
    ).astype(int)

    results["gnn_status"] = results[
        "gnn_anomaly"
    ].map({
        0: "Normal",
        1: "Anomaly"
    })

    return results


# ==================================================
# 8. LOAD ISOLATION FOREST
# ==================================================

@st.cache_resource
def load_isolation_forest():

    if not IF_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Isolation Forest model not found: {IF_MODEL_FILE}"
        )

    if not IF_THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"Isolation Forest threshold not found: "
            f"{IF_THRESHOLD_FILE}"
        )

    model = joblib.load(IF_MODEL_FILE)

    with open(IF_THRESHOLD_FILE, "r") as file:
        threshold_data = json.load(file)

    threshold = float(
        threshold_data["threshold"]
    )

    return model, threshold


# ==================================================
# 9. ISOLATION FOREST LIVE PREDICTION
# ==================================================

def predict_isolation_forest(
    model,
    threshold,
    live_features
):

    feature_data = live_features[
        FEATURE_COLUMNS
    ]

    feature_array = feature_data.to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(feature_array).all():
        raise ValueError(
            "Live features contain invalid values."
        )

    # Keep DataFrame column names because the model
    # was trained using a DataFrame.

    scores = model.decision_function(
        feature_data
    )

    results = live_features[
        ["location_id", "location_name", "time"]
    ].copy()

    results["if_anomaly_score"] = scores

    results["if_anomaly"] = (
        scores <= threshold
    ).astype(int)

    results["if_status"] = results[
        "if_anomaly"
    ].map({
        0: "Normal",
        1: "Anomaly"
    })

    return results


# ==================================================
# 10. DASHBOARD HEADER
# ==================================================

st.title(
    "🌦️ AI-Based Spatio-Temporal Tracking "
    "of Extreme Weather Anomalies"
)

st.markdown(
    """
    **Proof-of-concept dashboard**

    Historical weather analysis, live weather monitoring,
    graph-based anomaly detection, event tracking,
    and conditional diffusion.
    """
)

st.info(
    "Anomaly detections are statistical model outputs. "
    "They are not automatically confirmed extreme weather events."
)


# ==================================================
# 11. PROJECT OVERVIEW
# ==================================================

st.header("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Monitoring Locations", "5")
col2.metric("Historical Period", "2018–2020")
col3.metric("Anomaly Models", "GNN + Isolation Forest")
col4.metric("Weather Field Demo", "Conditional Diffusion")

st.divider()


# ==================================================
# 12. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🌐 Navigation")

section = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Dashboard Overview",
        "Live Weather",
        "GNN Anomaly Detection",
        "Model Comparison",
        "Tracked Weather Events",
        "Global City Monitor"
    ]
)


# ==================================================
# 13. DASHBOARD OVERVIEW
# ==================================================

if section == "Dashboard Overview":

    st.header("Welcome")

    st.write(
        "The project monitors five coastal locations "
        "in Andhra Pradesh:"
    )

    for location in LOCATION_NAMES:
        st.write(f"📍 {location}")

    st.success(
        "Dashboard foundation loaded successfully!"
    )


# ==================================================
# 14. LIVE WEATHER SECTION
# ==================================================

elif section == "Live Weather":

    st.header("🌤️ Live Weather Monitoring")

    raw_data = load_raw_weather()

    if raw_data is None:

        st.error(
            "Raw live weather data could not be loaded."
        )

    else:

        latest_time = raw_data["time"].max()

        st.caption(
            f"Latest data timestamp: {latest_time}"
        )

        st.caption(
            "These are API/model-based weather estimates, "
            "not necessarily direct station measurements."
        )

        selected_location = st.selectbox(
            "Choose a location",
            raw_data["location_name"].tolist()
        )

        selected_data = raw_data[
            raw_data["location_name"] == selected_location
        ].iloc[0]

        st.subheader(
            f"Weather Conditions — {selected_location}"
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "🌡️ Temperature",
            f"{selected_data['temperature_2m_c']:.2f} °C"
        )

        metric2.metric(
            "💧 Relative Humidity",
            f"{selected_data['relative_humidity_2m_pct']:.1f} %"
        )

        metric3.metric(
            "🌧️ Rainfall",
            f"{selected_data['precipitation_mm']:.2f} mm"
        )

        metric4, metric5, metric6 = st.columns(3)

        metric4.metric(
            "💨 Wind Speed",
            f"{selected_data['wind_speed_10m_kmh']:.2f} km/h"
        )

        metric5.metric(
            "🌬️ Wind Gusts",
            f"{selected_data['wind_gusts_10m_kmh']:.2f} km/h"
        )

        metric6.metric(
            "🔵 Pressure",
            f"{selected_data['pressure_msl_hpa']:.2f} hPa"
        )

        st.divider()

        st.subheader(
            "📍 Weather Across All Five Locations"
        )

        display_columns = [
            "location_name",
            "time",
            "temperature_2m_c",
            "relative_humidity_2m_pct",
            "precipitation_mm",
            "wind_speed_10m_kmh",
            "wind_gusts_10m_kmh",
            "pressure_msl_hpa"
        ]

        display_data = raw_data[
            display_columns
        ].rename(
            columns={
                "location_name": "Location",
                "time": "Timestamp",
                "temperature_2m_c": "Temperature (°C)",
                "relative_humidity_2m_pct": "Humidity (%)",
                "precipitation_mm": "Rainfall (mm)",
                "wind_speed_10m_kmh": "Wind Speed (km/h)",
                "wind_gusts_10m_kmh": "Wind Gusts (km/h)",
                "pressure_msl_hpa": "Pressure (hPa)"
            }
        )

        st.dataframe(
            display_data,
            width="stretch",
            hide_index=True
        )

        st.success(
            f"Successfully loaded readings for "
            f"{len(raw_data)} locations."
        )

        model_features = load_model_features()

        if model_features is not None:

            st.success(
                "Standardized live model features are available."
            )

            st.caption(
                f"Model feature records available: "
                f"{len(model_features)}"
            )

        else:

            st.warning(
                "Standardized live model features are unavailable."
            )


# ==================================================
# 15. GNN ANOMALY DETECTION
# ==================================================

elif section == "GNN Anomaly Detection":

    st.header("🤖 GNN-Based Weather Anomaly Detection")

    st.write(
        """
        The trained Spatial Graph Autoencoder reconstructs
        weather features for the five connected locations.

        A higher reconstruction error indicates that the
        observed weather pattern is less familiar to the model.
        """
    )

    try:

        model, adjacency, threshold = load_gnn_model()

        live_features = load_model_features()

        if live_features is None:

            st.error(
                "Live standardized features were not found. "
                "Run the live weather preprocessing script first."
            )

        else:

            results = predict_gnn_anomalies(
                model,
                adjacency,
                live_features,
                threshold
            )

            st.subheader("Model Information")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Monitoring Locations",
                len(results)
            )

            col2.metric(
                "Anomaly Threshold",
                f"{threshold:.6f}"
            )

            col3.metric(
                "Detected Anomalies",
                int(results["gnn_anomaly"].sum())
            )

            st.caption(
                "Threshold: 99th percentile of validation "
                "node reconstruction errors."
            )

            st.divider()

            st.subheader(
                "📍 Live Anomaly Results by Location"
            )

            display_results = results[
                [
                    "location_name",
                    "time",
                    "gnn_anomaly_score",
                    "gnn_status"
                ]
            ].rename(
                columns={
                    "location_name": "Location",
                    "time": "Timestamp",
                    "gnn_anomaly_score": "GNN Anomaly Score",
                    "gnn_status": "Detection Status"
                }
            )

            st.dataframe(
                display_results,
                width="stretch",
                hide_index=True
            )

            st.subheader("Location-wise Detection Status")

            columns = st.columns(5)

            for index, row in results.iterrows():

                with columns[index]:

                    st.markdown(
                        f"**{row['location_name']}**"
                    )

                    st.metric(
                        "Anomaly Score",
                        f"{row['gnn_anomaly_score']:.6f}"
                    )

                    if row["gnn_anomaly"] == 1:
                        st.error("⚠️ Anomaly")
                    else:
                        st.success("Normal")

            st.divider()

            st.subheader(
                "📊 GNN Reconstruction Error Comparison"
            )

            chart_data = results.set_index(
                "location_name"
            )["gnn_anomaly_score"]

            st.bar_chart(chart_data)

            st.caption(
                "Higher reconstruction error means greater "
                "deviation from patterns learned by the GNN."
            )

            anomaly_count = int(
                results["gnn_anomaly"].sum()
            )

            if anomaly_count > 0:

                st.warning(
                    f"The GNN flagged {anomaly_count} "
                    "location(s) for statistical review."
                )

            else:

                st.success(
                    "No monitored location exceeded the "
                    "current GNN anomaly threshold."
                )

            st.info(
                "These are statistical detections, not confirmed "
                "meteorological extreme events. The model evaluates "
                "the current spatial snapshot; it does not forecast "
                "future anomalies."
            )

    except Exception as error:

        st.error(
            f"GNN detection could not be completed: {error}"
        )

        st.write(
            "Check the GNN model, threshold, graph adjacency "
            "and live feature files."
        )


# ==================================================
# 16. GNN AND ISOLATION FOREST COMPARISON
# ==================================================

elif section == "Model Comparison":

    st.header("⚖️ GNN vs Isolation Forest")

    st.write(
        """
        This section evaluates both trained anomaly-detection
        models using the same standardized live weather inputs.

        The models use different scoring methods and separate
        validation-derived thresholds.
        """
    )

    try:

        live_features = load_model_features()

        if live_features is None:

            st.error(
                "Live standardized features are unavailable. "
                "Run the live preprocessing script first."
            )

        else:

            # ------------------------------------------
            # LOAD BOTH MODELS
            # ------------------------------------------

            gnn_model, adjacency, gnn_threshold = (
                load_gnn_model()
            )

            if_model, if_threshold = (
                load_isolation_forest()
            )

            # ------------------------------------------
            # RUN BOTH MODELS
            # ------------------------------------------

            gnn_results = predict_gnn_anomalies(
                gnn_model,
                adjacency,
                live_features,
                gnn_threshold
            )

            if_results = predict_isolation_forest(
                if_model,
                if_threshold,
                live_features
            )

            # ------------------------------------------
            # COMBINE RESULTS
            # ------------------------------------------

            comparison = pd.merge(
                gnn_results,
                if_results,
                on=[
                    "location_id",
                    "location_name",
                    "time"
                ],
                how="inner",
                validate="one_to_one"
            )

            if len(comparison) != 5:
                raise ValueError(
                    "Expected exactly five matched location "
                    "results from both models."
                )

            comparison["agreement"] = np.where(
                comparison["gnn_anomaly"]
                == comparison["if_anomaly"],
                "Agree",
                "Disagree"
            )

            comparison["combined_status"] = np.where(
                comparison["agreement"] == "Agree",
                comparison["gnn_status"],
                "Model Disagreement"
            )

            # ------------------------------------------
            # MODEL SUMMARY
            # ------------------------------------------

            st.subheader("Model Information")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### GNN Autoencoder")

                st.metric(
                    "GNN Threshold",
                    f"{gnn_threshold:.6f}"
                )

                st.metric(
                    "GNN Anomalies",
                    int(comparison["gnn_anomaly"].sum())
                )

                st.caption(
                    "Flags scores greater than the threshold."
                )

            with col2:

                st.markdown("### Isolation Forest")

                st.metric(
                    "Isolation Forest Threshold",
                    f"{if_threshold:.6f}"
                )

                st.metric(
                    "Isolation Forest Anomalies",
                    int(comparison["if_anomaly"].sum())
                )

                st.caption(
                    "Flags decision-function scores at or below "
                    "the threshold."
                )

            st.divider()

            # ------------------------------------------
            # AGREEMENT SUMMARY
            # ------------------------------------------

            st.subheader("Model Agreement")

            agreement_count = int(
                (comparison["agreement"] == "Agree").sum()
            )

            disagreement_count = (
                len(comparison) - agreement_count
            )

            agreement_percentage = (
                agreement_count / len(comparison)
            ) * 100

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Locations Compared",
                len(comparison)
            )

            col2.metric(
                "Models Agree",
                agreement_count
            )

            col3.metric(
                "Agreement",
                f"{agreement_percentage:.1f}%"
            )

            st.write(
                f"Model disagreements: {disagreement_count}"
            )

            # ------------------------------------------
            # COMPARISON TABLE
            # ------------------------------------------

            st.divider()

            st.subheader(
                "📍 Location-wise Model Comparison"
            )

            display_comparison = comparison[
                [
                    "location_name",
                    "time",
                    "gnn_anomaly_score",
                    "gnn_status",
                    "if_anomaly_score",
                    "if_status",
                    "agreement"
                ]
            ].rename(
                columns={
                    "location_name": "Location",
                    "time": "Timestamp",
                    "gnn_anomaly_score": "GNN Score",
                    "gnn_status": "GNN Status",
                    "if_anomaly_score": "Isolation Forest Score",
                    "if_status": "Isolation Forest Status",
                    "agreement": "Agreement"
                }
            )

            st.dataframe(
                display_comparison,
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------
            # DETECTION CHART
            # ------------------------------------------

            st.divider()

            st.subheader(
                "📊 Anomaly Detection Comparison"
            )

            chart_data = comparison.set_index(
                "location_name"
            )[
                [
                    "gnn_anomaly",
                    "if_anomaly"
                ]
            ].rename(
                columns={
                    "gnn_anomaly": "GNN",
                    "if_anomaly": "Isolation Forest"
                }
            )

            st.bar_chart(chart_data)

            st.caption(
                "1 = anomaly detected; 0 = normal. "
                "This chart compares binary decisions, not raw scores."
            )

            # ------------------------------------------
            # INTERPRETATION
            # ------------------------------------------

            st.divider()

            st.subheader("Interpretation")

            if disagreement_count == 0:

                st.success(
                    "Both models agree on the detection status "
                    "of all five monitored locations."
                )

            else:

                st.warning(
                    f"The models disagree on "
                    f"{disagreement_count} location(s). "
                    "These locations may need further examination."
                )

            st.info(
                "The GNN and Isolation Forest use different scoring "
                "scales. Their raw anomaly scores should not be "
                "compared directly. Agreement is not equivalent "
                "to accuracy because verified ground-truth event "
                "labels are not available here."
            )

            # ------------------------------------------
            # SAVE COMPARISON RESULTS
            # ------------------------------------------

            output_folder = (
                PROJECT_ROOT / "outputs" / "anomaly_results"
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True
            )

            output_file = (
                output_folder / "live_model_comparison.csv"
            )

            comparison.to_csv(
                output_file,
                index=False
            )

            st.success(
                "Live model comparison completed successfully."
            )

            st.caption(
                f"Comparison results saved to: {output_file}"
            )

    except Exception as error:

        st.error(
            f"Model comparison could not be completed: {error}"
        )

        st.write(
            "Check that both trained model files, both threshold "
            "files, the graph adjacency file and standardized "
            "live features exist."
        )




# ==================================================
# 17. TRACKED WEATHER EVENTS
# ==================================================

elif section == "Tracked Weather Events":

    st.header("🧭 Spatio-Temporal Weather Event Tracking")

    st.write(
        "This section groups GNN anomaly detections across time "
        "and connected locations into candidate weather events."
    )

    st.info(
        "These groups are candidate events derived from model "
        "detections. They are not independently verified "
        "meteorological events. Severity categories and grouping "
        "rules are proof-of-concept settings."
    )

    events_file = (
        PROJECT_ROOT / "outputs" / "anomaly_results"
        / "tracked_weather_events.csv"
    )

    if not events_file.exists():

        st.error(
            "Tracked events file was not found. Run "
            "backend/event_tracker.py first."
        )
        st.caption(f"Expected file: {events_file}")

    else:

        try:

            events = pd.read_csv(events_file)

            required_event_columns = [
                "event_id",
                "start_time",
                "end_time",
                "duration_hours",
                "anomaly_records",
                "distinct_timestamps",
                "number_of_locations",
                "location_ids",
                "locations",
                "maximum_gnn_anomaly_score",
                "severity",
                "dominant_weather_signal",
                "event_category",
                "event_status",
                "event_type"
            ]

            missing_event_columns = [
                column for column in required_event_columns
                if column not in events.columns
            ]

            if missing_event_columns:
                raise ValueError(
                    "The tracked events CSV is missing expected "
                    f"columns: {missing_event_columns}"
                )

            if events.empty:

                st.warning(
                    "The tracked events file exists but contains "
                    "no event records."
                )

            else:

                events["start_time"] = pd.to_datetime(
                    events["start_time"], errors="coerce"
                )
                events["end_time"] = pd.to_datetime(
                    events["end_time"], errors="coerce"
                )
                events["duration_hours"] = pd.to_numeric(
                    events["duration_hours"], errors="coerce"
                )
                events["anomaly_records"] = pd.to_numeric(
                    events["anomaly_records"], errors="coerce"
                )
                events["number_of_locations"] = pd.to_numeric(
                    events["number_of_locations"], errors="coerce"
                )
                events["maximum_gnn_anomaly_score"] = pd.to_numeric(
                    events["maximum_gnn_anomaly_score"],
                    errors="coerce"
                )

                events = events.dropna(
                    subset=["start_time", "event_id"]
                ).copy()

                if events.empty:
                    st.warning(
                        "No valid event rows remain after checking "
                        "the event IDs and start timestamps."
                    )
                else:

                    # ------------------------------------------
                    # SUMMARY METRICS
                    # ------------------------------------------

                    st.subheader("Event Summary")

                    total_events = len(events)
                    multi_record_events = int(
                        (
                            events["event_category"]
                            == "Multi-record candidate event"
                        ).sum()
                    )
                    isolated_events = int(
                        (
                            events["event_category"]
                            == "Isolated detection"
                        ).sum()
                    )
                    locations_involved = int(
                        events["number_of_locations"]
                        .fillna(0).max()
                    )

                    metric1, metric2, metric3, metric4 = (
                        st.columns(4)
                    )

                    metric1.metric(
                        "Tracked Groups", total_events
                    )
                    metric2.metric(
                        "Multi-record Groups",
                        multi_record_events
                    )
                    metric3.metric(
                        "Isolated Detections",
                        isolated_events
                    )
                    metric4.metric(
                        "Max Locations in a Group",
                        locations_involved
                    )

                    st.caption(
                        "A multi-record group contains multiple "
                        "anomaly records under the configured time "
                        "window and graph-connectivity rules."
                    )

                    st.divider()

                    # ------------------------------------------
                    # FILTERS
                    # ------------------------------------------

                    st.subheader("Explore Candidate Events")

                    filter_col1, filter_col2, filter_col3 = (
                        st.columns(3)
                    )

                    severity_values = sorted(
                        events["severity"].dropna().astype(str).unique()
                    )
                    category_values = sorted(
                        events["event_category"]
                        .dropna().astype(str).unique()
                    )
                    signal_values = sorted(
                        events["dominant_weather_signal"]
                        .dropna().astype(str).unique()
                    )

                    with filter_col1:
                        selected_severities = st.multiselect(
                            "Filter by severity",
                            severity_values,
                            default=severity_values
                        )

                    with filter_col2:
                        selected_categories = st.multiselect(
                            "Filter by event category",
                            category_values,
                            default=category_values
                        )

                    with filter_col3:
                        selected_signals = st.multiselect(
                            "Filter by dominant signal",
                            signal_values,
                            default=signal_values
                        )

                    filtered_events = events[
                        events["severity"].astype(str).isin(
                            selected_severities
                        )
                        & events["event_category"].astype(str).isin(
                            selected_categories
                        )
                        & events[
                            "dominant_weather_signal"
                        ].astype(str).isin(selected_signals)
                    ].copy()

                    st.write(
                        f"Showing **{len(filtered_events)}** of "
                        f"**{len(events)}** tracked groups."
                    )

                    # ------------------------------------------
                    # EVENTS TABLE
                    # ------------------------------------------

                    table_columns = [
                        "event_id",
                        "start_time",
                        "end_time",
                        "duration_hours",
                        "anomaly_records",
                        "number_of_locations",
                        "locations",
                        "severity",
                        "dominant_weather_signal",
                        "event_category",
                        "event_status",
                        "event_type"
                    ]

                    display_events = filtered_events[
                        table_columns
                    ].rename(
                        columns={
                            "event_id": "Event ID",
                            "start_time": "Start Time",
                            "end_time": "End Time",
                            "duration_hours": "Duration (hours)",
                            "anomaly_records": "Anomaly Records",
                            "number_of_locations": "Locations Count",
                            "locations": "Locations",
                            "severity": "Severity",
                            "dominant_weather_signal": "Dominant Signal",
                            "event_category": "Event Category",
                            "event_status": "Event Status",
                            "event_type": "Event Type"
                        }
                    )

                    st.dataframe(
                        display_events,
                        width="stretch",
                        hide_index=True
                    )

                    # ------------------------------------------
                    # EVENT DISTRIBUTIONS
                    # ------------------------------------------

                    st.divider()
                    st.subheader("Candidate Event Distributions")

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        st.markdown("**Groups by severity**")
                        severity_chart = (
                            filtered_events["severity"]
                            .value_counts()
                        )
                        if not severity_chart.empty:
                            st.bar_chart(severity_chart)
                        else:
                            st.caption(
                                "No events match the current filters."
                            )

                    with chart_col2:
                        st.markdown("**Groups by dominant signal**")
                        signal_chart = (
                            filtered_events[
                                "dominant_weather_signal"
                            ].value_counts()
                        )
                        if not signal_chart.empty:
                            st.bar_chart(signal_chart)
                        else:
                            st.caption(
                                "No events match the current filters."
                            )

                    # ------------------------------------------
                    # SELECTED EVENT DETAILS
                    # ------------------------------------------

                    st.divider()
                    st.subheader("Inspect One Candidate Event")

                    if not filtered_events.empty:

                        event_options = (
                            filtered_events["event_id"]
                            .astype(str).tolist()
                        )

                        selected_event_id = st.selectbox(
                            "Choose an event ID",
                            event_options
                        )

                        selected_event = filtered_events[
                            filtered_events["event_id"].astype(str)
                            == selected_event_id
                        ].iloc[0]

                        detail_col1, detail_col2, detail_col3 = (
                            st.columns(3)
                        )

                        detail_col1.metric(
                            "Anomaly Records",
                            int(
                                selected_event["anomaly_records"]
                                if pd.notna(
                                    selected_event["anomaly_records"]
                                ) else 0
                            )
                        )

                        detail_col2.metric(
                            "Distinct Timestamps",
                            int(
                                selected_event["distinct_timestamps"]
                                if pd.notna(
                                    selected_event["distinct_timestamps"]
                                ) else 0
                            )
                        )

                        detail_col3.metric(
                            "Maximum GNN Score",
                            (
                                f"{selected_event['maximum_gnn_anomaly_score']:.6f}"
                                if pd.notna(
                                    selected_event[
                                        "maximum_gnn_anomaly_score"
                                    ]
                                )
                                else "N/A"
                            )
                        )

                        st.write(
                            f"**Time span:** "
                            f"{selected_event['start_time']} to "
                            f"{selected_event['end_time']}"
                        )
                        st.write(
                            f"**Locations:** "
                            f"{selected_event['locations']}"
                        )
                        st.write(
                            f"**Dominant weather signal:** "
                            f"{selected_event['dominant_weather_signal']}"
                        )
                        st.write(
                            f"**Category:** "
                            f"{selected_event['event_category']}"
                        )
                        st.write(
                            f"**Status / type:** "
                            f"{selected_event['event_status']} / "
                            f"{selected_event['event_type']}"
                        )

                    else:
                        st.warning(
                            "No events match the selected filters. "
                            "Change one or more filters to inspect events."
                        )

                    st.caption(
                        "The dominant signal is the feature with the "
                        "largest absolute standardized value in the "
                        "group. It is a descriptive indicator, not "
                        "evidence of causation. Severity bins are "
                        "configured for this proof-of-concept and are "
                        "not meteorologically calibrated."
                    )

        except Exception as error:

            st.error(
                f"Tracked weather events could not be displayed: {error}"
            )
            st.write(
                "Confirm that backend/event_tracker.py has completed "
                "successfully and that the output CSV uses the expected "
                "15-column schema."
            )





# ==================================================
# 18. GLOBAL CITY WEATHER ANOMALY MONITOR
# ==================================================

elif section == "Global City Monitor":

    st.header("🌍 Global City Weather Anomaly Monitor")

    st.write(
        "Monitor the configured global city watchlist using live "
        "Open-Meteo weather estimates and the trained Isolation Forest "
        "and experimental six-node GNN inference."
    )

    st.warning(
        "Model-indicated anomaly levels are statistical screening outputs, "
        "not official weather warnings or confirmed extreme events. "
        "The models were trained on five Andhra Pradesh locations. "
        "Applying them to distant cities is experimental and affected by "
        "geographical and data-distribution shift."
    )

    backend_folder = PROJECT_ROOT / "backend"
    watchlist_file = (
        PROJECT_ROOT / "datasets" / "processed"
        / "global_city_watchlist.json"
    )
    monitor_output_file = (
        PROJECT_ROOT / "outputs" / "global_city_monitor"
        / "global_city_anomaly_results.csv"
    )
    monitor_script = backend_folder / "global_city_monitor.py"

    # Add-city workflow. The watchlist module uses the same tested
    # Open-Meteo geocoding helper as the standalone monitor scripts.
    st.subheader("➕ Add a city to the watchlist")
    with st.form("add_global_monitor_city", clear_on_submit=True):
        city_to_add = st.text_input(
            "City name", placeholder="For example: Hyderabad, Toronto, Nairobi"
        )
        add_city_submitted = st.form_submit_button("Add city")

    if add_city_submitted:
        if not city_to_add.strip():
            st.warning("Enter a city name first.")
        else:
            try:
                backend_path_text = str(backend_folder)
                if backend_path_text not in sys.path:
                    sys.path.insert(0, backend_path_text)
                from global_city_watchlist import add_city

                add_result = add_city(city_to_add.strip())
                if add_result is False:
                    st.warning(
                        "The city was not added. It may already be in the "
                        "watchlist or could not be resolved."
                    )
                else:
                    st.success(
                        f"Processed '{city_to_add.strip()}'. "
                        "The watchlist has been updated if the city was new."
                    )
                    st.session_state["global_city_force_scan"] = True
                    st.rerun()
            except Exception as error:
                st.error(f"Could not add the city: {error}")

    st.divider()
    st.subheader("⚙️ Monitoring and refresh settings")
    refresh_col1, refresh_col2 = st.columns([1, 2])
    with refresh_col1:
        refresh_minutes = st.selectbox(
            "Automatic scan interval",
            options=[5, 10, 15, 30],
            index=0,
            format_func=lambda value: f"Every {value} minutes"
        )
    with refresh_col2:
        st.caption(
            "The scan runs when this page is opened and repeats while the "
            "dashboard remains open. Use the button below to scan immediately. "
            "Each scan requests current weather for the watchlist and the five "
            "reference locations; avoid setting an unnecessarily short interval."
        )

    def run_global_city_scan():
        if not monitor_script.exists():
            raise FileNotFoundError(
                f"Monitor script not found: {monitor_script}"
            )
        completed = subprocess.run(
            [sys.executable, str(monitor_script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(details or "Global city scan failed.")
        return completed.stdout

    @st.fragment(run_every=refresh_minutes * 60)
    def global_city_results_fragment():
        manual_scan = st.button(
            "🔄 Scan cities now", type="primary", key="global_city_manual_scan"
        )
        force_scan = st.session_state.pop("global_city_force_scan", False)
        file_is_missing = not monitor_output_file.exists()
        file_is_stale = (
            monitor_output_file.exists()
            and (datetime.now().timestamp() - monitor_output_file.stat().st_mtime)
            >= refresh_minutes * 60
        )
        should_scan = manual_scan or force_scan or file_is_missing or file_is_stale

        if should_scan:
            with st.spinner("Fetching live weather and running both anomaly models..."):
                try:
                    scan_log = run_global_city_scan()
                    st.session_state["global_city_last_scan"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    st.session_state["global_city_scan_log"] = scan_log
                    st.success("Global city scan completed.")
                except Exception as error:
                    st.error(f"Global city scan could not be completed: {error}")

        if monitor_output_file.exists():
            try:
                results = pd.read_csv(monitor_output_file)
                if results.empty:
                    st.warning("The monitor output exists but contains no city results.")
                    return

                last_scan = st.session_state.get("global_city_last_scan")
                if last_scan:
                    st.caption(f"Last scan completed by this dashboard: {last_scan}")
                else:
                    modified = datetime.fromtimestamp(
                        monitor_output_file.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    st.caption(f"Latest saved scan file timestamp: {modified}")

                status_column = next(
                    (column for column in ["combined_status", "status"]
                     if column in results.columns),
                    None
                )
                if status_column:
                    statuses = results[status_column].astype(str)
                    high_count = int(statuses.str.contains("High Model-Indicated", case=False).sum())
                    elevated_count = int(statuses.str.contains("Elevated Model-Indicated", case=False).sum())
                    normal_count = int(statuses.str.contains("No Anomaly Flagged", case=False).sum())
                    metric1, metric2, metric3, metric4 = st.columns(4)
                    metric1.metric("Cities checked", len(results))
                    metric2.metric("High model-indicated", high_count)
                    metric3.metric("Elevated model-indicated", elevated_count)
                    metric4.metric("No anomaly flagged", normal_count)

                # Sort by monitor risk rank when present. The original CSV
                # ordering is retained when no rank column is available.
                if "risk_rank" in results.columns:
                    results = results.sort_values(
                        ["risk_rank"], ascending=False
                    )

                st.subheader("📍 Ranked city results")
                st.dataframe(
                    results,
                    width="stretch",
                    hide_index=True
                )

                if status_column:
                    high_rows = results[
                        results[status_column].astype(str).str.contains(
                            "High Model-Indicated", case=False
                        )
                    ]
                    elevated_rows = results[
                        results[status_column].astype(str).str.contains(
                            "Elevated Model-Indicated", case=False
                        )
                    ]
                    if not high_rows.empty:
                        st.error(
                            "High model-indicated anomaly result(s): "
                            + ", ".join(
                                high_rows.get("city", high_rows.iloc[:, 0]).astype(str).tolist()
                            )
                        )
                    elif not elevated_rows.empty:
                        st.warning(
                            "No city is currently in the highest combined category. "
                            "Elevated model-indicated results are present."
                        )
                    else:
                        st.success("No monitored city was flagged by the combined status rule.")

                with st.expander("View latest scan log"):
                    log_text = st.session_state.get("global_city_scan_log")
                    if log_text:
                        st.code(log_text, language="text")
                    else:
                        st.caption(
                            "The CSV was loaded from a previous scan. Press 'Scan cities now' "
                            "to display a fresh run log."
                        )

                st.caption(f"Results file: {monitor_output_file}")
            except Exception as error:
                st.error(f"Could not read or display monitor results: {error}")
        else:
            st.info("No scan results are saved yet. Start a scan to populate the city table.")

    global_city_results_fragment()

    st.divider()
    st.caption(
        "Coverage is the finite city watchlist, not every city worldwide. "
        "Add cities above to expand it. Weather values are Open-Meteo API/model "
        "estimates; anomaly flags are not official hazard alerts."
    )


# ==================================================
# 19. REFRESH DASHBOARD
# ==================================================

st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Dashboard"):

    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
