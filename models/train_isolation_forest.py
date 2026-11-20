
import pandas as pd
import numpy as np
import joblib
import json

from pathlib import Path
from sklearn.ensemble import IsolationForest

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORICAL_FILE = (
    PROJECT_ROOT / "datasets" / "processed"
    / "historical_features.csv"
)

LIVE_FILE = (
    PROJECT_ROOT / "datasets" / "raw"
    / "live_features.csv"
)

MODEL_FOLDER = PROJECT_ROOT / "models"

OUTPUT_FOLDER = (
    PROJECT_ROOT / "outputs" / "anomaly_results"
)

MODEL_FILE = MODEL_FOLDER / "isolation_forest.pkl"
THRESHOLD_FILE = MODEL_FOLDER / "if_threshold.json"

# The eight model input features
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

# Percentage of validation observations used
# to establish the demonstration threshold
THRESHOLD_PERCENTILE = 1.0


def load_data():

    print("Loading processed historical data...")

    historical_df = pd.read_csv(
        HISTORICAL_FILE,
        parse_dates=["time"]
    )

    live_df = pd.read_csv(
        LIVE_FILE,
        parse_dates=["time"]
    )

    historical_df = historical_df.sort_values(
        ["time", "location_id"]
    ).reset_index(drop=True)

    live_df = live_df.sort_values(
        ["time", "location_id"]
    ).reset_index(drop=True)

    return historical_df, live_df


def chronological_split(df):

    # Split by timestamp, not individual rows.
    # This keeps all five locations at a timestamp
    # together in the same partition.

    unique_times = np.sort(
        df["time"].unique()
    )

    train_end = int(len(unique_times) * 0.70)
    validation_end = int(len(unique_times) * 0.85)

    train_times = unique_times[:train_end]
    validation_times = unique_times[train_end:validation_end]
    test_times = unique_times[validation_end:]

    train_df = df[
        df["time"].isin(train_times)
    ].copy()

    validation_df = df[
        df["time"].isin(validation_times)
    ].copy()

    test_df = df[
        df["time"].isin(test_times)
    ].copy()

    return train_df, validation_df, test_df


def train_model(train_df):

    print("\nTraining Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        train_df[FEATURE_COLUMNS]
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("Isolation Forest trained successfully.")
    print("Model saved:", MODEL_FILE)

    return model


def calculate_threshold(model, validation_df):

    print("\nCalculating validation threshold...")

    validation_scores = model.decision_function(
        validation_df[FEATURE_COLUMNS]
    )

    threshold = float(
        np.percentile(
            validation_scores,
            THRESHOLD_PERCENTILE
        )
    )

    threshold_info = {
        "threshold": threshold,
        "validation_percentile": THRESHOLD_PERCENTILE,
        "description": (
            "Observations with decision_function "
            "scores at or below this value are flagged."
        )
    }

    with open(THRESHOLD_FILE, "w") as file:
        json.dump(
            threshold_info,
            file,
            indent=4
        )

    print("Threshold:", threshold)

    return threshold


def detect_anomalies(model, df, threshold):

    result = df.copy()

    result["anomaly_score"] = (
        model.decision_function(
            result[FEATURE_COLUMNS]
        )
    )

    result["anomaly"] = (
        result["anomaly_score"] <= threshold
    ).astype(int)

    result["anomaly_label"] = result["anomaly"].map(
        {
            0: "Normal",
            1: "Anomaly"
        }
    )

    return result


def main():

    MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    historical_df, live_df = load_data()

    print("\nHistorical dataset:", historical_df.shape)
    print("Live dataset:", live_df.shape)

    # Create chronological partitions
    train_df, validation_df, test_df = (
        chronological_split(historical_df)
    )

    print("\nChronological data split:")
    print("Training records:", len(train_df))
    print("Validation records:", len(validation_df))
    print("Test records:", len(test_df))

    # Train model using training period only
    model = train_model(train_df)

    # Determine threshold using validation period
    threshold = calculate_threshold(
        model,
        validation_df
    )

    # Score historical records for retrospective analysis
    historical_results = detect_anomalies(
        model,
        historical_df,
        threshold
    )

    # Score current live records using same model
    live_results = detect_anomalies(
        model,
        live_df,
        threshold
    )

    # Save historical results
    historical_results.to_csv(
        OUTPUT_FOLDER / "historical_anomalies.csv",
        index=False
    )

    # Save live results
    live_results.to_csv(
        OUTPUT_FOLDER / "live_anomalies.csv",
        index=False
    )

    # Display summaries
    print("\nHistorical anomaly summary:")

    print(
        historical_results["anomaly_label"]
        .value_counts()
    )

    print("\nAnomalies by location:")

    print(
        historical_results.groupby(
            "location_name"
        )["anomaly"].sum()
    )

    print("\nLive weather anomaly results:")

    print(
        live_results[
            [
                "location_name",
                "time",
                "anomaly_score",
                "anomaly_label"
            ]
        ].to_string(index=False)
    )

    print("\nAnomaly detection completed!")

    print("Results saved in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
