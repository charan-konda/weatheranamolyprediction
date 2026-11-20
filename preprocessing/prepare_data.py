
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORICAL_FILE = (
    PROJECT_ROOT / "datasets" / "processed"
    / "andhra_pradesh_weather_cleaned.csv"
)

LIVE_FILE = (
    PROJECT_ROOT / "datasets" / "raw"
    / "live_weather.csv"
)

MODEL_FOLDER = PROJECT_ROOT / "models"

SCALER_FILE = MODEL_FOLDER / "weather_scaler.pkl"

# Weather features used by the ML models
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


def add_wind_features(df):
    """Convert circular wind direction into numerical features."""

    direction_radians = np.deg2rad(
        df["wind_direction_10m_deg"]
    )

    df["wind_sin"] = np.sin(direction_radians)
    df["wind_cos"] = np.cos(direction_radians)

    return df


def load_historical_data():

    print("Loading historical dataset...")

    df = pd.read_csv(
        HISTORICAL_FILE,
        parse_dates=["time"]
    )

    df = df.sort_values(
        ["time", "location_id"]
    ).reset_index(drop=True)

    df = add_wind_features(df)

    print("Historical data loaded:", df.shape)

    return df


def load_live_data():

    print("Loading live weather dataset...")

    df = pd.read_csv(
        LIVE_FILE,
        parse_dates=["time"]
    )

    df = df.sort_values(
        ["time", "location_id"]
    ).reset_index(drop=True)

    df = add_wind_features(df)

    print("Live data loaded:", df.shape)

    return df


def train_scaler(historical_df):

    print("\nFitting feature scaler...")

    # Create chronological split
    unique_times = np.sort(
        historical_df["time"].unique()
    )

    split_index = int(len(unique_times) * 0.70)

    training_times = unique_times[:split_index]

    training_df = historical_df[
        historical_df["time"].isin(training_times)
    ]

    scaler = StandardScaler()

    scaler.fit(training_df[FEATURE_COLUMNS])

    # Save scaler for future inference
    MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(scaler, SCALER_FILE)

    print("Scaler trained successfully.")
    print("Scaler saved at:", SCALER_FILE)

    return scaler


def transform_data(df, scaler):

    transformed_values = scaler.transform(
        df[FEATURE_COLUMNS]
    )

    transformed_df = pd.DataFrame(
        transformed_values,
        columns=FEATURE_COLUMNS,
        index=df.index
    )

    # Preserve location and timestamp information
    result = pd.concat(
        [
            df[["location_id", "location_name", "time"]],
            transformed_df
        ],
        axis=1
    )

    return result


if __name__ == "__main__":

    historical_df = load_historical_data()

    live_df = load_live_data()

    scaler = train_scaler(historical_df)

    historical_processed = transform_data(
        historical_df,
        scaler
    )

    live_processed = transform_data(
        live_df,
        scaler
    )

    # Save processed datasets
    historical_output = (
        PROJECT_ROOT / "datasets" / "processed"
        / "historical_features.csv"
    )

    live_output = (
        PROJECT_ROOT / "datasets" / "raw"
        / "live_features.csv"
    )

    historical_processed.to_csv(
        historical_output,
        index=False
    )

    live_processed.to_csv(
        live_output,
        index=False
    )

    print("\nPreprocessing completed!")

    print(
        "Historical feature shape:",
        historical_processed.shape
    )

    print(
        "Live feature shape:",
        live_processed.shape
    )

    print("\nProcessed feature preview:")
    print(historical_processed.head())
