
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from city_weather import (
    search_city,
    fetch_current_weather
)


# -------------------------------------------------
# 1. PROJECT CONFIGURATION
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FOLDER = PROJECT_ROOT / "models"

SCALER_FILE = MODEL_FOLDER / "weather_scaler.pkl"
IF_MODEL_FILE = MODEL_FOLDER / "isolation_forest.pkl"
IF_THRESHOLD_FILE = MODEL_FOLDER / "if_threshold.json"

GNN_MODEL_FILE = MODEL_FOLDER / "gnn_autoencoder.pth"
GNN_THRESHOLD_FILE = MODEL_FOLDER / "gnn_threshold.json"

DEVICE = torch.device("cpu")

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

REFERENCE_LOCATIONS = [
    {
        "location_name": "Visakhapatnam",
        "latitude": 17.6868,
        "longitude": 83.2185
    },
    {
        "location_name": "Kakinada",
        "latitude": 16.9891,
        "longitude": 82.2475
    },
    {
        "location_name": "Machilipatnam",
        "latitude": 16.1809,
        "longitude": 81.1303
    },
    {
        "location_name": "Nellore",
        "latitude": 14.4426,
        "longitude": 79.9865
    },
    {
        "location_name": "Ongole",
        "latitude": 15.5057,
        "longitude": 80.0499
    }
]

HIDDEN_DIM = 32
K_NEIGHBORS = 2


# -------------------------------------------------
# 2. GNN ARCHITECTURE
# Must match models/train_gnn.py
# -------------------------------------------------

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


# -------------------------------------------------
# 3. LOAD TRAINED MODELS
# -------------------------------------------------

def load_models():

    required_files = [
        SCALER_FILE,
        IF_MODEL_FILE,
        IF_THRESHOLD_FILE,
        GNN_MODEL_FILE,
        GNN_THRESHOLD_FILE
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required model files are missing:\n"
            + "\n".join(missing_files)
        )

    scaler = joblib.load(SCALER_FILE)

    isolation_forest = joblib.load(IF_MODEL_FILE)

    with open(IF_THRESHOLD_FILE, "r") as file:
        if_threshold = float(
            json.load(file)["threshold"]
        )

    with open(GNN_THRESHOLD_FILE, "r") as file:
        gnn_threshold = float(
            json.load(file)["threshold"]
        )

    checkpoint = torch.load(
        GNN_MODEL_FILE,
        map_location=DEVICE,
        weights_only=True
    )

    # Verify that the saved model uses the expected
    # feature ordering and dimensions.
    if checkpoint["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError(
            "GNN checkpoint feature columns do not "
            "match the current feature configuration."
        )

    if checkpoint["input_dim"] != len(FEATURE_COLUMNS):
        raise ValueError(
            "Unexpected GNN input dimension."
        )

    gnn_model = SpatialGraphAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    ).to(DEVICE)

    gnn_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    gnn_model.eval()

    return {
        "scaler": scaler,
        "isolation_forest": isolation_forest,
        "if_threshold": if_threshold,
        "gnn_model": gnn_model,
        "gnn_threshold": gnn_threshold
    }


# -------------------------------------------------
# 4. CONVERT WEATHER INTO MODEL FEATURES
# -------------------------------------------------

def prepare_weather_features(weather):

    direction_radians = np.deg2rad(
        weather["wind_direction_10m_deg"]
    )

    feature_values = {
        "temperature_2m_c":
            weather["temperature_2m_c"],

        "relative_humidity_2m_pct":
            weather["relative_humidity_2m_pct"],

        "precipitation_mm":
            weather["precipitation_mm"],

        "wind_speed_10m_kmh":
            weather["wind_speed_10m_kmh"],

        "pressure_msl_hpa":
            weather["pressure_msl_hpa"],

        "wind_gusts_10m_kmh":
            weather["wind_gusts_10m_kmh"],

        "wind_sin":
            np.sin(direction_radians),

        "wind_cos":
            np.cos(direction_radians)
    }

    feature_df = pd.DataFrame(
        [feature_values],
        columns=FEATURE_COLUMNS
    )

    if not np.isfinite(
        feature_df.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Weather data contains missing or "
            "non-finite feature values."
        )

    return feature_df


# -------------------------------------------------
# 5. BUILD SIX-NODE GEOGRAPHIC GRAPH
# -------------------------------------------------

def calculate_distance(lat1, lon1, lat2, lon2):

    earth_radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return earth_radius * c


def build_city_graph(city):

    locations = REFERENCE_LOCATIONS + [
        {
            "location_name": city["city"],
            "latitude": city["latitude"],
            "longitude": city["longitude"]
        }
    ]

    number_of_nodes = len(locations)

    distances = np.zeros(
        (number_of_nodes, number_of_nodes),
        dtype=np.float32
    )

    for i in range(number_of_nodes):

        for j in range(number_of_nodes):

            distances[i, j] = calculate_distance(
                locations[i]["latitude"],
                locations[i]["longitude"],
                locations[j]["latitude"],
                locations[j]["longitude"]
            )

    # Reproduce the original graph construction:
    # K nearest neighbours, then symmetrization.

    adjacency = np.zeros_like(distances)

    for i in range(number_of_nodes):

        nearest_nodes = np.argsort(
            distances[i]
        )[1:K_NEIGHBORS + 1]

        for j in nearest_nodes:
            adjacency[i, j] = 1.0

    adjacency = np.maximum(
        adjacency,
        adjacency.T
    )

    # Recalculate Gaussian distance weights
    # for the six-node graph.

    positive_distances = distances[
        distances > 0
    ]

    sigma = np.mean(positive_distances)

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            "Unable to calculate graph distance scale."
        )

    weights = np.exp(
        -(distances ** 2) / (2 * sigma ** 2)
    )

    weighted_adjacency = adjacency * weights

    # Match the training-time GNN normalization:
    # add self-loops, then symmetric normalization.

    adjacency_with_self_loops = (
        weighted_adjacency
        + np.eye(number_of_nodes, dtype=np.float32)
    )

    degrees = adjacency_with_self_loops.sum(axis=1)

    degree_inverse_sqrt = np.diag(
        1.0 / np.sqrt(degrees)
    )

    normalized_adjacency = (
        degree_inverse_sqrt
        @ adjacency_with_self_loops
        @ degree_inverse_sqrt
    )

    return torch.tensor(
        normalized_adjacency,
        dtype=torch.float32,
        device=DEVICE
    )


# -------------------------------------------------
# 6. FETCH REFERENCE WEATHER
# -------------------------------------------------

def fetch_reference_weather():

    reference_features = []

    for location in REFERENCE_LOCATIONS:

        print(
            "Fetching reference weather:",
            location["location_name"]
        )

        weather = fetch_current_weather(
            location["latitude"],
            location["longitude"],
            timezone="auto"
        )

        feature_df = prepare_weather_features(
            weather
        )

        reference_features.append(
            feature_df.iloc[0].to_numpy(
                dtype=np.float32
            )
        )

    return np.stack(reference_features)


# -------------------------------------------------
# 7. RUN CITY ANOMALY INFERENCE
# -------------------------------------------------

def analyze_city(city):

    print("\nLoading trained models...")

    models = load_models()

    print("\nFetching selected city weather...")

    city_weather = fetch_current_weather(
        city["latitude"],
        city["longitude"],
        timezone=city.get("timezone", "auto")
    )

    city_features = prepare_weather_features(
        city_weather
    )

    # Transform using the original training scaler.
    city_scaled = models["scaler"].transform(
        city_features
    ).astype(np.float32)

    # -----------------------------
    # Isolation Forest inference
    # -----------------------------

    if_score = float(
        models["isolation_forest"].decision_function(
            city_scaled
        )[0]
    )

    if_anomaly = (
        if_score <= models["if_threshold"]
    )

    # -----------------------------
    # GNN experimental inference
    # -----------------------------

    print(
        "\nFetching five reference locations "
        "for experimental GNN inference..."
    )

    reference_raw = fetch_reference_weather()

    all_raw_features = np.vstack(
        [
            reference_raw,
            city_features.to_numpy(dtype=np.float32)
        ]
    )

    all_scaled_features = models["scaler"].transform(
        pd.DataFrame(
            all_raw_features,
            columns=FEATURE_COLUMNS
        )
    ).astype(np.float32)

    graph_adjacency = build_city_graph(city)

    input_tensor = torch.tensor(
        all_scaled_features[np.newaxis, :, :],
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        reconstructed = models["gnn_model"](
            input_tensor,
            graph_adjacency
        )

        node_errors = torch.mean(
            (input_tensor - reconstructed) ** 2,
            dim=2
        )

    # The selected city is node index 5.
    gnn_score = float(
        node_errors[0, 5].item()
    )

    gnn_anomaly = (
        gnn_score > models["gnn_threshold"]
    )

    # -----------------------------
    # Return structured results
    # -----------------------------

    result = {
        "city": city["city"],
        "country": city.get("country", ""),
        "region": city.get("admin1", ""),
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "weather_time": city_weather.get("time", "Unknown"),
        "weather": city_weather,

        "isolation_forest": {
            "score": if_score,
            "threshold": models["if_threshold"],
            "anomaly": bool(if_anomaly),
            "label": (
                "Anomaly" if if_anomaly else "Normal"
            )
        },

        "gnn": {
            "score": gnn_score,
            "threshold": models["gnn_threshold"],
            "anomaly": bool(gnn_anomaly),
            "label": (
                "Anomaly" if gnn_anomaly else "Normal"
            ),
            "status": "Experimental six-node transfer inference"
        },

        "interpretation": (
            "Model-indicated anomaly results only. "
            "These are not official weather warnings "
            "or validated forecasts."
        )
    }

    return result


# -------------------------------------------------
# 8. STANDALONE TEST
# -------------------------------------------------


def main():

    print("=" * 65)
    print("CITY-LEVEL AI WEATHER ANOMALY TEST")
    print("=" * 65)

    city_name = "Hyderabad"

    print("\nSearching for:", city_name)

    matches = search_city(city_name)

    if not matches:
        raise ValueError(
            "No matching city was found."
        )

    city_result = matches[0]

    # Adapt the geocoding result to the inference
    # module's expected city dictionary.
    city = {
        "city": city_result.get(
            "name",
            city_result.get("city", city_name)
        ),
        "country": city_result.get("country", ""),
        "admin1": city_result.get("admin1", ""),
        "latitude": city_result["latitude"],
        "longitude": city_result["longitude"],
        "timezone": city_result.get("timezone", "auto")
    }

    print(
        "Selected:",
        city["city"],
        city["admin1"],
        city["country"]
    )

    result = analyze_city(city)

    print("\n" + "=" * 65)
    print("CITY ANOMALY RESULTS")
    print("=" * 65)

    print("\nCity:", result["city"])
    print("Country:", result["country"])
    print("Region:", result["region"])
    print("Weather time:", result["weather_time"])

    print("\nISOLATION FOREST")
    print("-" * 40)

    print(
        "Anomaly score:",
        round(
            result["isolation_forest"]["score"],
            6
        )
    )

    print(
        "Threshold:",
        round(
            result["isolation_forest"]["threshold"],
            6
        )
    )

    print(
        "Result:",
        result["isolation_forest"]["label"]
    )

    print("\nGRAPH NEURAL NETWORK")
    print("-" * 40)

    print(
        "Anomaly score:",
        round(
            result["gnn"]["score"],
            6
        )
    )

    print(
        "Threshold:",
        round(
            result["gnn"]["threshold"],
            6
        )
    )

    print(
        "Result:",
        result["gnn"]["label"]
    )

    print(
        "Status:",
        result["gnn"]["status"]
    )

    print("\n" + result["interpretation"])

    print("\n" + "=" * 65)
    print("CITY MODEL INFERENCE TEST COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
