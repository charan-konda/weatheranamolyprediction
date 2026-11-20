
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


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

MODEL_FILE = (
    PROJECT_ROOT / "models" / "gnn_autoencoder.pth"
)

THRESHOLD_FILE = (
    PROJECT_ROOT / "models" / "gnn_threshold.json"
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
# 3. SAME GNN ARCHITECTURE USED DURING TRAINING
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

    data = data.sort_values(
        "location_id"
    ).reset_index(drop=True)

    return data


# ==================================================
# 6. LOAD TRAINED GNN AND GRAPH
# ==================================================

@st.cache_resource
def load_gnn_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"GNN model not found: {MODEL_FILE}"
        )

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph file not found: {GRAPH_FILE}"
        )

    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"GNN threshold not found: {THRESHOLD_FILE}"
        )

    # Load checkpoint created by train_gnn.py
    checkpoint = torch.load(
        MODEL_FILE,
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
            "Model input dimension does not match feature count."
        )

    model = SpatialGraphAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # Load graph in exactly the same location order
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

    # Add self-loops
    adjacency = (
        adjacency + np.eye(len(LOCATION_IDS))
    )

    # Same symmetric degree normalization as training
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

    # Load validation-derived anomaly threshold
    import json

    with open(THRESHOLD_FILE, "r") as file:
        threshold_data = json.load(file)

    threshold = float(
        threshold_data["threshold"]
    )

    return model, adjacency_tensor, threshold


# ==================================================
# 7. CALCULATE LIVE GNN ANOMALY SCORES
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
            "Live location names do not match the training order."
        )

    feature_array = live_features[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    if not np.isfinite(feature_array).all():
        raise ValueError(
            "Live model features contain missing or infinite values."
        )

    # Shape: [1, 5, 8]
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

        # Mean squared reconstruction error
        # Shape: [1, 5]
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

    results["status"] = results[
        "gnn_anomaly"
    ].map({
        0: "Normal",
        1: "Anomaly"
    })

    return results


# ==================================================
# 8. DASHBOARD HEADER
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
# 9. PROJECT OVERVIEW
# ==================================================

st.header("📊 Project Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Monitoring Locations", "5")
col2.metric("Historical Period", "2018–2020")
col3.metric("Anomaly Model", "GNN Autoencoder")
col4.metric("Weather Field Demo", "Conditional Diffusion")

st.divider()


# ==================================================
# 10. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🌐 Navigation")

section = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Dashboard Overview",
        "Live Weather",
        "GNN Anomaly Detection"
    ]
)


# ==================================================
# 11. DASHBOARD OVERVIEW
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
# 12. LIVE WEATHER SECTION
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
            "These are original API measurements, "
            "not standardized model features."
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
# 13. GNN ANOMALY DETECTION SECTION
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

        # Load model, graph and threshold
        model, adjacency, threshold = load_gnn_model()

        # Load standardized live input
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

            # ------------------------------------------
            # MODEL INFORMATION
            # ------------------------------------------

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

            # ------------------------------------------
            # ANOMALY RESULTS
            # ------------------------------------------

            st.subheader(
                "📍 Live Anomaly Results by Location"
            )

            display_results = results[
                [
                    "location_name",
                    "time",
                    "gnn_anomaly_score",
                    "status"
                ]
            ].copy()

            display_results = display_results.rename(
                columns={
                    "location_name": "Location",
                    "time": "Timestamp",
                    "gnn_anomaly_score": "GNN Anomaly Score",
                    "status": "Detection Status"
                }
            )

            st.dataframe(
                display_results,
                width="stretch",
                hide_index=True
            )

            # ------------------------------------------
            # ANOMALY STATUS CARDS
            # ------------------------------------------

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

            # ------------------------------------------
            # SCORE COMPARISON
            # ------------------------------------------

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

            # ------------------------------------------
            # INTERPRETATION
            # ------------------------------------------

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
                "These are model-based statistical detections, "
                "not confirmed meteorological extreme events. "
                "The model evaluates the current spatial snapshot; "
                "it does not forecast future anomalies."
            )

    except Exception as error:

        st.error(
            f"GNN detection could not be completed: {error}"
        )

        st.write(
            "Check that the trained model, threshold, "
            "graph adjacency and live feature files exist "
            "in their expected project folders."
        )


# ==================================================
# 14. REFRESH DASHBOARD
# ==================================================

st.sidebar.divider()

if st.sidebar.button("🔄 Refresh Dashboard"):

    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
