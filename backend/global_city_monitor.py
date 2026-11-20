"""
Global City Weather Monitoring Engine
-------------------------------------
Fetches recent hourly weather for the configured city watchlist,
builds a dynamic spatial graph, and runs:

1. Isolation Forest on each city's latest weather observation.
2. The trained Spatio-Temporal GNN on aligned 24-hour sequences.

IMPORTANT RESEARCH LIMITATION:
The first ST-GNN checkpoint was trained on five Andhra Pradesh cities.
Inference for other cities is experimental transfer inference and is
not validated global meteorological anomaly detection or an official
weather warning.
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_FOLDER = Path(__file__).resolve().parent

MODEL_FOLDER = PROJECT_ROOT / "models"
SCALER_FILE = MODEL_FOLDER / "weather_scaler.pkl"
IF_MODEL_FILE = MODEL_FOLDER / "isolation_forest.pkl"
IF_THRESHOLD_FILE = MODEL_FOLDER / "if_threshold.json"

STGNN_MODEL_FILE = MODEL_FOLDER / "stgnn_autoencoder.pth"
STGNN_THRESHOLD_FILE = MODEL_FOLDER / "stgnn_threshold.json"

GRAPH_FILE = (
    PROJECT_ROOT / "outputs" / "weather_graph"
    / "weighted_adjacency_matrix.csv"
)

OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "global_city_monitor"
OUTPUT_FILE = OUTPUT_FOLDER / "global_city_anomaly_results.csv"

SEQUENCE_LENGTH = 24
NEAREST_NEIGHBORS = 2
WEATHER_TIMEZONE = "GMT"
DEVICE = torch.device("cpu")

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

REFERENCE_LOCATIONS = [
    {
        "city": "Visakhapatnam",
        "country": "India",
        "latitude": 17.6868,
        "longitude": 83.2185,
        "timezone": "Asia/Kolkata",
    },
    {
        "city": "Kakinada",
        "country": "India",
        "latitude": 16.9891,
        "longitude": 82.2475,
        "timezone": "Asia/Kolkata",
    },
    {
        "city": "Machilipatnam",
        "country": "India",
        "latitude": 16.1809,
        "longitude": 81.1303,
        "timezone": "Asia/Kolkata",
    },
    {
        "city": "Nellore",
        "country": "India",
        "latitude": 14.4426,
        "longitude": 79.9865,
        "timezone": "Asia/Kolkata",
    },
    {
        "city": "Ongole",
        "country": "India",
        "latitude": 15.5057,
        "longitude": 80.0499,
        "timezone": "Asia/Kolkata",
    },
]

if str(BACKEND_FOLDER) not in sys.path:
    sys.path.insert(0, str(BACKEND_FOLDER))

from city_weather import (  # noqa: E402
    fetch_current_weather,
    fetch_recent_hourly_weather,
)
from global_city_watchlist import load_watchlist  # noqa: E402


# ---------------------------------------------------------
# 2. SPATIO-TEMPORAL GNN ARCHITECTURE
# Must match models/train_stgnn.py exactly.
# ---------------------------------------------------------

class SpatioTemporalGraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.self_projection = nn.Linear(input_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(input_dim, hidden_dim)

        self.spatial_refinement = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_refinement = nn.Linear(hidden_dim, hidden_dim)

        self.activation = nn.ReLU()

        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, adjacency):
        batch_size, sequence_length, node_count, feature_count = x.shape

        neighbor_features = torch.einsum(
            "nm,btmf->btnf",
            adjacency,
            x,
        )

        hidden = self.activation(
            self.self_projection(x)
            + self.neighbor_projection(neighbor_features)
        )

        neighbor_hidden = torch.einsum(
            "nm,btmh->btnh",
            adjacency,
            hidden,
        )

        hidden = self.activation(
            self.spatial_refinement(hidden)
            + self.neighbor_refinement(neighbor_hidden)
        )

        hidden = hidden.permute(0, 2, 1, 3).contiguous()
        hidden = hidden.view(
            batch_size * node_count,
            sequence_length,
            -1,
        )

        temporal_output, _ = self.temporal_gru(hidden)

        temporal_output = temporal_output.view(
            batch_size,
            node_count,
            sequence_length,
            -1,
        ).permute(0, 2, 1, 3).contiguous()

        return self.decoder(temporal_output)


# ---------------------------------------------------------
# 3. LOAD MODELS, SCALER, THRESHOLDS AND REFERENCE GRAPH
# ---------------------------------------------------------

def load_models():
    required_files = [
        SCALER_FILE,
        IF_MODEL_FILE,
        IF_THRESHOLD_FILE,
        STGNN_MODEL_FILE,
        STGNN_THRESHOLD_FILE,
        GRAPH_FILE,
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required project files are missing:\n" + "\n".join(missing)
        )

    print("Loading scaler, Isolation Forest and ST-GNN...")

    scaler = joblib.load(SCALER_FILE)
    isolation_forest = joblib.load(IF_MODEL_FILE)

    with open(IF_THRESHOLD_FILE, "r", encoding="utf-8") as file:
        if_threshold = float(json.load(file)["threshold"])

    with open(STGNN_THRESHOLD_FILE, "r", encoding="utf-8") as file:
        threshold_data = json.load(file)
        stgnn_threshold = float(threshold_data["threshold"])
        saved_sequence_length = int(
            threshold_data.get("sequence_length_hours", SEQUENCE_LENGTH)
        )

    if saved_sequence_length != SEQUENCE_LENGTH:
        raise ValueError(
            f"ST-GNN expects {saved_sequence_length} hours, "
            f"but this monitor is configured for {SEQUENCE_LENGTH}."
        )

    checkpoint = torch.load(
        STGNN_MODEL_FILE,
        map_location=DEVICE,
        weights_only=True,
    )

    checkpoint_features = checkpoint.get("feature_columns")
    if checkpoint_features and list(checkpoint_features) != FEATURE_COLUMNS:
        raise ValueError(
            "ST-GNN checkpoint feature order does not match this monitor."
        )

    input_dim = int(checkpoint["input_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])

    if input_dim != len(FEATURE_COLUMNS):
        raise ValueError(
            f"ST-GNN expects {input_dim} features, "
            f"but {len(FEATURE_COLUMNS)} are configured."
        )

    stgnn_model = SpatioTemporalGraphAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(DEVICE)

    stgnn_model.load_state_dict(checkpoint["model_state_dict"])
    stgnn_model.eval()

    graph_df = pd.read_csv(GRAPH_FILE, index_col=0)
    reference_names = [item["city"] for item in REFERENCE_LOCATIONS]

    try:
        graph_df = graph_df.loc[reference_names, reference_names]
    except KeyError as error:
        raise ValueError(
            "The saved reference adjacency matrix does not contain "
            "all five expected reference cities."
        ) from error

    original_adjacency = graph_df.to_numpy(dtype=np.float32)

    if original_adjacency.shape != (5, 5):
        raise ValueError("The reference adjacency matrix must be 5 x 5.")

    if not np.isfinite(original_adjacency).all():
        raise ValueError("The reference adjacency matrix has invalid values.")

    return {
        "scaler": scaler,
        "isolation_forest": isolation_forest,
        "if_threshold": if_threshold,
        "stgnn_model": stgnn_model,
        "stgnn_threshold": stgnn_threshold,
        "original_adjacency": original_adjacency,
    }


# ---------------------------------------------------------
# 4. WEATHER FEATURE VALIDATION AND SCALING
# ---------------------------------------------------------

def validate_feature_row(weather):
    missing = [column for column in FEATURE_COLUMNS if column not in weather]
    if missing:
        raise ValueError(
            "Weather record is missing ST-GNN features: "
            + ", ".join(missing)
        )

    values = np.asarray(
        [weather[column] for column in FEATURE_COLUMNS],
        dtype=np.float32,
    )

    if not np.isfinite(values).all():
        raise ValueError("Weather record contains missing or invalid features.")

    return {column: float(weather[column]) for column in FEATURE_COLUMNS}


def standardize_rows(rows, scaler):
    frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    scaled = scaler.transform(frame)

    if not np.isfinite(scaled).all():
        raise ValueError("Standardized weather features are not finite.")

    return scaled.astype(np.float32)


def weather_to_feature_row(weather):
    """
    Convert the current-weather dictionary to the eight model features.
    The updated city_weather.py supplies wind_sin and wind_cos directly.
    """
    if "wind_sin" in weather and "wind_cos" in weather:
        return validate_feature_row(weather)

    required = [
        "temperature_2m_c",
        "relative_humidity_2m_pct",
        "precipitation_mm",
        "wind_speed_10m_kmh",
        "wind_direction_10m_deg",
        "pressure_msl_hpa",
        "wind_gusts_10m_kmh",
    ]
    missing = [key for key in required if key not in weather]
    if missing:
        raise ValueError(
            "Current weather is missing fields: " + ", ".join(missing)
        )

    direction = math.radians(float(weather["wind_direction_10m_deg"]))

    return validate_feature_row(
        {
            "temperature_2m_c": float(weather["temperature_2m_c"]),
            "relative_humidity_2m_pct": float(
                weather["relative_humidity_2m_pct"]
            ),
            "precipitation_mm": float(weather["precipitation_mm"]),
            "wind_speed_10m_kmh": float(weather["wind_speed_10m_kmh"]),
            "pressure_msl_hpa": float(weather["pressure_msl_hpa"]),
            "wind_gusts_10m_kmh": float(weather["wind_gusts_10m_kmh"]),
            "wind_sin": math.sin(direction),
            "wind_cos": math.cos(direction),
        }
    )


# ---------------------------------------------------------
# 5. DYNAMIC GRAPH FOR FIVE REFERENCES + WATCHLIST CITIES
# ---------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    radius_km = 6371.0

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2],
    )

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2.0) ** 2
    )

    value = min(1.0, max(0.0, value))
    return 2.0 * radius_km * math.asin(math.sqrt(value))


def build_dynamic_adjacency(cities, original_adjacency):
    """
    Keep the original five-node reference graph weights.
    Add each watchlist city as a node and connect it to its
    nearest geographic nodes. New edges use a distance kernel.

    Geographic proximity is only a simple graph construction choice;
    it does not model atmospheric transport.
    """
    all_locations = REFERENCE_LOCATIONS + cities
    node_count = len(all_locations)

    if node_count < 2:
        raise ValueError("At least two graph nodes are required.")

    coordinates = []
    for location in all_locations:
        if "latitude" not in location or "longitude" not in location:
            raise ValueError(
                f"Coordinates are missing for {location.get('city', 'Unknown')}."
            )
        coordinates.append(
            (float(location["latitude"]), float(location["longitude"]))
        )

    distances = np.zeros((node_count, node_count), dtype=np.float64)

    for i in range(node_count):
        for j in range(i + 1, node_count):
            distance = haversine_distance(
                coordinates[i][0],
                coordinates[i][1],
                coordinates[j][0],
                coordinates[j][1],
            )
            distances[i, j] = distance
            distances[j, i] = distance

    adjacency = np.zeros((node_count, node_count), dtype=np.float32)

    # Preserve the trained reference graph exactly before normalization.
    adjacency[:5, :5] = original_adjacency

    # Estimate a distance scale from all non-zero pairwise distances.
    upper_distances = distances[np.triu_indices(node_count, k=1)]
    positive_distances = upper_distances[upper_distances > 0]

    if positive_distances.size == 0:
        raise ValueError("All graph nodes have identical coordinates.")

    sigma = float(np.mean(positive_distances))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Could not calculate a valid graph distance scale.")

    # Add spatial links for each watchlist city to its nearest nodes.
    for node_index in range(5, node_count):
        candidates = [
            index for index in range(node_count)
            if index != node_index
        ]
        nearest = sorted(
            candidates,
            key=lambda index: distances[node_index, index],
        )[:min(NEAREST_NEIGHBORS, len(candidates))]

        for neighbor_index in nearest:
            distance = distances[node_index, neighbor_index]
            weight = math.exp(-(distance ** 2) / (2.0 * sigma ** 2))
            adjacency[node_index, neighbor_index] = weight
            adjacency[neighbor_index, node_index] = weight

    adjacency += np.eye(node_count, dtype=np.float32)

    degrees = adjacency.sum(axis=1)
    inverse_sqrt = np.diag(
        1.0 / np.sqrt(np.maximum(degrees, 1e-12))
    )

    normalized = inverse_sqrt @ adjacency @ inverse_sqrt

    if not np.isfinite(normalized).all():
        raise ValueError("Dynamic graph normalization produced invalid values.")

    return normalized.astype(np.float32)


# ---------------------------------------------------------
# 6. FETCH ALIGNED 24-HOUR SEQUENCES
# ---------------------------------------------------------

def fetch_hourly_sequence(location):
    """
    Fetch 24 hourly records in GMT so all cities share aligned timestamps.
    """
    return fetch_recent_hourly_weather(
        float(location["latitude"]),
        float(location["longitude"]),
        timezone=WEATHER_TIMEZONE,
        hours=SEQUENCE_LENGTH,
    )


def prepare_all_sequences(cities):
    """
    Fetch the five reference sequences and all watchlist sequences.
    A single failed city/reference fetch stops the scan instead of
    silently scoring a graph with missing nodes.
    """
    all_locations = REFERENCE_LOCATIONS + cities
    sequence_by_city = {}

    print(
        f"\nFetching aligned {SEQUENCE_LENGTH}-hour weather sequences "
        f"for {len(all_locations)} graph nodes..."
    )

    for index, location in enumerate(all_locations, start=1):
        name = location.get("city", "Unknown")
        country = location.get("country", "")

        print(f"  [{index}/{len(all_locations)}] {name}, {country}")

        rows = fetch_hourly_sequence(location)

        if len(rows) != SEQUENCE_LENGTH:
            raise RuntimeError(
                f"{name}: expected {SEQUENCE_LENGTH} hourly rows, "
                f"received {len(rows)}."
            )

        times = [row["time"] for row in rows]
        if len(set(times)) != SEQUENCE_LENGTH:
            raise RuntimeError(
                f"{name}: hourly timestamps contain duplicates."
            )

        sequence_by_city[(name.casefold(), country.casefold())] = rows

    # Ensure every city uses exactly the same UTC timestamps.
    reference_times = [
        row["time"]
        for row in sequence_by_city[
            (
                REFERENCE_LOCATIONS[0]["city"].casefold(),
                REFERENCE_LOCATIONS[0]["country"].casefold(),
            )
        ]
    ]

    for location in all_locations[1:]:
        key = (
            location.get("city", "").casefold(),
            location.get("country", "").casefold(),
        )
        city_times = [row["time"] for row in sequence_by_city[key]]

        if city_times != reference_times:
            raise RuntimeError(
                f"Hourly timestamps for {location.get('city')} do not "
                "align with the other graph nodes. The scan was stopped "
                "to avoid mixing different time steps."
            )

    return sequence_by_city, reference_times


# ---------------------------------------------------------
# 7. RUN SPATIO-TEMPORAL INFERENCE ON THE FULL GRAPH
# ---------------------------------------------------------

def run_stgnn_inference(cities, models):
    sequence_by_city, common_times = prepare_all_sequences(cities)

    all_locations = REFERENCE_LOCATIONS + cities
    node_sequences = []

    for location in all_locations:
        key = (
            location.get("city", "").casefold(),
            location.get("country", "").casefold(),
        )
        rows = sequence_by_city[key]

        # Apply the same scaler and feature order used in training.
        raw_rows = [
            validate_feature_row(row)
            for row in rows
        ]
        scaled = standardize_rows(raw_rows, models["scaler"])
        node_sequences.append(scaled)

    # [time, nodes, features]
    graph_sequence = np.stack(node_sequences, axis=1)

    if graph_sequence.shape != (
        SEQUENCE_LENGTH,
        len(all_locations),
        len(FEATURE_COLUMNS),
    ):
        raise ValueError(
            f"Unexpected ST-GNN input shape: {graph_sequence.shape}"
        )

    adjacency = build_dynamic_adjacency(
        cities,
        models["original_adjacency"],
    )

    input_tensor = torch.tensor(
        graph_sequence[np.newaxis, :, :, :],
        dtype=torch.float32,
        device=DEVICE,
    )
    adjacency_tensor = torch.tensor(
        adjacency,
        dtype=torch.float32,
        device=DEVICE,
    )

    models["stgnn_model"].eval()
    with torch.no_grad():
        reconstructed = models["stgnn_model"](
            input_tensor,
            adjacency_tensor,
        )

        # Score the most recent hour, matching training threshold definition.
        latest_actual = input_tensor[:, -1, :, :]
        latest_reconstructed = reconstructed[:, -1, :, :]

        node_errors = torch.mean(
            (latest_actual - latest_reconstructed) ** 2,
            dim=2,
        )

    scores = node_errors.cpu().numpy()[0]

    watchlist_scores = {}
    for index, city in enumerate(cities, start=5):
        key = (
            city.get("city", "").casefold(),
            city.get("country", "").casefold(),
        )
        watchlist_scores[key] = float(scores[index])

    return watchlist_scores, common_times[-1], adjacency.shape


# ---------------------------------------------------------
# 8. ISOLATION FOREST CURRENT-OBSERVATION SCREENING
# ---------------------------------------------------------

def analyze_isolation_forest(city, current_weather, models):
    feature_row = weather_to_feature_row(current_weather)

    scaled = standardize_rows(
        [feature_row],
        models["scaler"],
    )

    feature_frame = pd.DataFrame(
        scaled,
        columns=FEATURE_COLUMNS,
    )

    score = float(
        models["isolation_forest"].decision_function(feature_frame)[0]
    )
    flagged = int(score <= models["if_threshold"])

    return score, flagged


# ---------------------------------------------------------
# 9. COMBINE MODEL RESULTS FOR ONE CITY
# ---------------------------------------------------------

def build_city_result(city, current_weather, if_score, if_anomaly, stgnn_score, models):
    stgnn_anomaly = int(stgnn_score > models["stgnn_threshold"])

    # This is a simple two-model vote, not a calibrated risk probability.
    if if_anomaly and stgnn_anomaly:
        status = "High Model-Indicated Anomaly"
        risk_rank = 2
    elif if_anomaly or stgnn_anomaly:
        status = "Elevated Model-Indicated Anomaly"
        risk_rank = 1
    else:
        status = "No Anomaly Flagged"
        risk_rank = 0

    return {
        "city": city.get("city", ""),
        "country": city.get("country", ""),
        "region": city.get("admin1", ""),
        "weather_time": current_weather.get("time", ""),
        "temperature_c": current_weather["temperature_2m_c"],
        "humidity_pct": current_weather["relative_humidity_2m_pct"],
        "rainfall_mm": current_weather["precipitation_mm"],
        "wind_speed_kmh": current_weather["wind_speed_10m_kmh"],
        "wind_gusts_kmh": current_weather["wind_gusts_10m_kmh"],
        "pressure_hpa": current_weather["pressure_msl_hpa"],
        "if_score": if_score,
        "if_threshold": models["if_threshold"],
        "if_status": "Anomaly" if if_anomaly else "Normal",
        "gnn_score": stgnn_score,
        "gnn_threshold": models["stgnn_threshold"],
        "gnn_status": "Anomaly" if stgnn_anomaly else "Normal",
        "model_indicated_status": status,
        "risk_rank": risk_rank,
        "gnn_note": (
            "Experimental 24-hour ST-GNN dynamic-graph transfer inference; "
            "trained on five Andhra Pradesh reference locations"
        ),
    }


# ---------------------------------------------------------
# 10. RUN THE GLOBAL WATCHLIST SCAN
# ---------------------------------------------------------

def run_global_scan():
    print("\n" + "=" * 68)
    print("GLOBAL CITY SPATIO-TEMPORAL WEATHER ANOMALY MONITOR")
    print("=" * 68)

    cities = load_watchlist()

    if not cities:
        raise ValueError(
            "The city watchlist is empty. Add at least one city."
        )

    # Validate and de-duplicate watchlist entries.
    unique_cities = []
    seen = set()

    for city in cities:
        name = str(city.get("city", "")).strip()
        country = str(city.get("country", "")).strip()

        if not name:
            print("Skipping a watchlist entry with no city name.")
            continue

        key = (name.casefold(), country.casefold())
        if key in seen:
            print(f"Skipping duplicate watchlist entry: {name}, {country}")
            continue
        seen.add(key)

        if city.get("latitude") is None or city.get("longitude") is None:
            raise ValueError(
                f"{name}, {country} has no coordinates. "
                "Resolve it through the city search/watchlist first."
            )

        unique_cities.append(city)

    cities = unique_cities

    if not cities:
        raise ValueError("No valid cities remain in the watchlist.")

    print(f"\nCities in watchlist: {len(cities)}")
    models = load_models()

    # ST-GNN sequence inference uses one aligned graph containing
    # all five training reference nodes and all configured watchlist cities.
    stgnn_scores, stgnn_time, graph_shape = run_stgnn_inference(
        cities,
        models,
    )

    print("\nST-GNN graph shape:", graph_shape)
    print("ST-GNN latest aligned hourly timestamp (GMT):", stgnn_time)

    results = []
    errors = []

    print("\nFetching current readings and running Isolation Forest...")

    for index, city in enumerate(cities, start=1):
        city_name = city.get("city", "Unknown")
        country = city.get("country", "")

        print(
            f"\n[{index}/{len(cities)}] "
            f"Checking {city_name}, {country}"
        )

        try:
            current_weather = fetch_current_weather(
                float(city["latitude"]),
                float(city["longitude"]),
                city.get("timezone", "auto"),
            )

            if_score, if_anomaly = analyze_isolation_forest(
                city,
                current_weather,
                models,
            )

            city_key = (
                city_name.casefold(),
                country.casefold(),
            )
            stgnn_score = stgnn_scores[city_key]

            result = build_city_result(
                city,
                current_weather,
                if_score,
                if_anomaly,
                stgnn_score,
                models,
            )
            results.append(result)

            print(
                "  Isolation Forest:",
                result["if_status"],
                f"({result['if_score']:.6f})",
            )
            print(
                "  ST-GNN:",
                result["gnn_status"],
                f"({result['gnn_score']:.6f})",
            )
            print("  Combined status:", result["model_indicated_status"])

        except Exception as error:
            message = f"{city_name}, {country}: {error}"
            errors.append(message)
            print("  CITY CHECK FAILED:", error)

    if results:
        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            ["risk_rank", "gnn_score"],
            ascending=[False, False],
        ).reset_index(drop=True)

        results_df.insert(
            0,
            "monitoring_rank",
            range(1, len(results_df) + 1),
        )

        results_df["scan_time"] = datetime.now().isoformat(
            timespec="seconds"
        )

        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(OUTPUT_FILE, index=False)

        print("\n" + "=" * 68)
        print("GLOBAL MONITORING SUMMARY")
        print("=" * 68)
        print("Successfully checked:", len(results_df))
        print("Failed city checks:", len(errors))
        print(
            "High model-indicated:",
            int((results_df["risk_rank"] == 2).sum()),
        )
        print(
            "Elevated model-indicated:",
            int((results_df["risk_rank"] == 1).sum()),
        )
        print(
            "No anomaly flagged:",
            int((results_df["risk_rank"] == 0).sum()),
        )

        print("\nRanked city results:")
        for _, row in results_df.iterrows():
            print(
                f"{row['monitoring_rank']}. "
                f"{row['city']}, {row['country']} — "
                f"{row['model_indicated_status']} "
                f"(ST-GNN score: {row['gnn_score']:.6f})"
            )

        print("\nResults saved to:")
        print(OUTPUT_FILE)
    else:
        print("\nNo city results were produced. Existing output was not replaced.")

    if errors:
        print("\nCITY CHECK ERRORS")
        for error in errors:
            print("-", error)

    print("\nGLOBAL CITY MONITOR SCAN COMPLETED")
    return results


if __name__ == "__main__":
    try:
        run_global_scan()
    except Exception as error:
        print("\nGLOBAL MONITOR FAILED")
        print(error)
        raise
