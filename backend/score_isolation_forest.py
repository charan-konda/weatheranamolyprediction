
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT / "models" / "isolation_forest.pkl"
)

THRESHOLD_FILE = (
    PROJECT_ROOT / "models" / "if_threshold.json"
)

LIVE_FILE = (
    PROJECT_ROOT / "datasets" / "raw"
    / "live_features.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT / "outputs" / "anomaly_results"
)

OUTPUT_FILE = (
    OUTPUT_FOLDER / "isolation_forest_live_results.csv"
)


# ==========================================
# 2. MODEL FEATURES
# ==========================================

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


# ==========================================
# 3. LOAD LIVE FEATURES
# ==========================================

def load_live_features():

    print("Loading standardized live weather features...")

    if not LIVE_FILE.exists():
        raise FileNotFoundError(
            f"Live feature file not found: {LIVE_FILE}"
        )

    data = pd.read_csv(
        LIVE_FILE,
        parse_dates=["time"]
    )

    if data.empty:
        raise ValueError(
            "Live feature file is empty."
        )

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
            f"Missing required columns: {missing}"
        )

    data = data.sort_values(
        "location_id"
    ).reset_index(drop=True)

    if len(data) != 5:
        raise ValueError(
            f"Expected five live location records, "
            f"but found {len(data)}."
        )

    feature_values = data[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    if not np.isfinite(feature_values).all():
        raise ValueError(
            "Live features contain missing or infinite values."
        )

    return data


# ==========================================
# 4. LOAD TRAINED ISOLATION FOREST
# ==========================================

def load_model_and_threshold():

    print("Loading trained Isolation Forest...")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_FILE}"
        )

    model = joblib.load(MODEL_FILE)

    with open(THRESHOLD_FILE, "r") as file:
        threshold_data = json.load(file)

    threshold = float(
        threshold_data["threshold"]
    )

    print("Model loaded successfully.")
    print("Anomaly threshold:", threshold)

    return model, threshold


# ==========================================
# 5. DETECT LIVE ANOMALIES
# ==========================================

def detect_live_anomalies(
    model,
    threshold,
    live_data
):

    print("Calculating Isolation Forest scores...")

    result = live_data[
        [
            "location_id",
            "location_name",
            "time"
        ]
    ].copy()

    features = live_data[
        FEATURE_COLUMNS
    ]

    # Isolation Forest decision function:
    # Lower values indicate more unusual observations.

    scores = model.decision_function(features)

    result["if_anomaly_score"] = scores

    result["if_anomaly"] = (
        scores <= threshold
    ).astype(int)

    result["if_status"] = result[
        "if_anomaly"
    ].map({
        0: "Normal",
        1: "Anomaly"
    })

    return result


# ==========================================
# 6. SAVE AND DISPLAY RESULTS
# ==========================================

def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    live_data = load_live_features()

    model, threshold = load_model_and_threshold()

    results = detect_live_anomalies(
        model,
        threshold,
        live_data
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nISOLATION FOREST LIVE RESULTS")
    print("=" * 60)

    print(
        results[
            [
                "location_name",
                "time",
                "if_anomaly_score",
                "if_status"
            ]
        ].to_string(index=False)
    )

    print("\nAnomaly summary:")

    print(
        results["if_status"].value_counts()
    )

    print("\nResults saved successfully:")
    print(OUTPUT_FILE)

    print("\nIsolation Forest live detection completed!")


if __name__ == "__main__":
    main()
