
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "outputs/anomaly_results/gnn_historical_anomalies.csv"
)

GRAPH_FILE = (
    ROOT / "outputs/weather_graph/adjacency_matrix.csv"
)

THRESHOLD_FILE = ROOT / "models/gnn_threshold.json"

FEATURE_FILE = (
    ROOT / "datasets/processed/historical_features.csv"
)

OUTPUT_DIR = ROOT / "outputs/anomaly_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_FILE = OUTPUT_DIR / "tracked_weather_events.csv"
ISOLATED_FILE = OUTPUT_DIR / "isolated_weather_anomalies.csv"

TIME_WINDOW_HOURS = 3

LOCATION_NAMES = {
    0: "Visakhapatnam",
    1: "Kakinada",
    2: "Machilipatnam",
    3: "Nellore",
    4: "Ongole"
}

# These features are standardized using the saved
# historical training scaler.
WEATHER_FEATURES = [
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "pressure_msl_hpa",
    "wind_gusts_10m_kmh",
    "wind_sin",
    "wind_cos"
]

FEATURE_LABELS = {
    "temperature_2m_c": "Temperature",
    "relative_humidity_2m_pct": "Humidity",
    "precipitation_mm": "Rainfall",
    "wind_speed_10m_kmh": "Wind speed",
    "pressure_msl_hpa": "Atmospheric pressure",
    "wind_gusts_10m_kmh": "Wind gusts",
    "wind_sin": "Wind direction",
    "wind_cos": "Wind direction"
}


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print("Loading GNN anomaly results...")

df = pd.read_csv(INPUT_FILE)
df["time"] = pd.to_datetime(df["time"])

required_columns = [
    "time",
    "location_id",
    "location_name",
    "gnn_anomaly_score",
    "gnn_anomaly"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns in GNN results: {missing}"
    )

df["location_id"] = df["location_id"].astype(int)

df = df.sort_values(
    ["time", "location_id"]
).reset_index(drop=True)

anomalies = df[
    df["gnn_anomaly"].astype(bool)
].copy()

print("Total historical records:", len(df))
print("Anomaly records:", len(anomalies))


# --------------------------------------------------
# 3. Load GNN threshold
# --------------------------------------------------

with open(THRESHOLD_FILE, "r") as file:
    threshold_data = json.load(file)

gnn_threshold = float(threshold_data["threshold"])

print("GNN anomaly threshold:", gnn_threshold)


# --------------------------------------------------
# 4. Load geographic graph
# --------------------------------------------------

print("\nLoading geographic graph...")

adjacency = pd.read_csv(
    GRAPH_FILE,
    index_col=0
).values

if adjacency.shape != (5, 5):
    raise ValueError(
        "Expected a 5 x 5 geographic adjacency matrix."
    )

connected_locations = {
    location_id: set()
    for location_id in LOCATION_NAMES
}

for i in range(5):
    for j in range(5):

        if i != j and adjacency[i, j] != 0:
            connected_locations[i].add(j)


# --------------------------------------------------

# --------------------------------------------------
# 5. Load standardized weather features
# --------------------------------------------------

print("\nLoading standardized weather features...")

features_df = pd.read_csv(FEATURE_FILE)

features_df["time"] = pd.to_datetime(
    features_df["time"]
)

feature_columns = [
    col for col in WEATHER_FEATURES
    if col in features_df.columns
]

if not feature_columns:
    raise ValueError(
        "No expected weather feature columns found. "
        f"Available columns: {features_df.columns.tolist()}"
    )

# Keep only the identifying columns and weather features.
features_df = features_df[
    ["time", "location_id"] + feature_columns
].copy()

# Remove any overlapping feature columns from anomalies
# before merging, preventing Pandas suffixes such as _x/_y.
anomalies = anomalies.drop(
    columns=[
        col for col in feature_columns
        if col in anomalies.columns
    ],
    errors="ignore"
)

# Join standardized features to anomaly records.
anomalies = anomalies.merge(
    features_df,
    on=["time", "location_id"],
    how="left",
    validate="one_to_one"
)

missing_features = anomalies[
    feature_columns
].isna().any(axis=1).sum()

if missing_features > 0:
    raise ValueError(
        f"{missing_features} anomaly records have missing "
        "weather features after merging."
    )

print("Weather features successfully joined.")
print("Features used:", feature_columns)


# --------------------------------------------------
# 6. Identify dominant weather signal
# --------------------------------------------------

def identify_weather_signal(row):

    # Features are standardized. The largest absolute
    # value indicates the strongest standardized deviation.
    values = row[feature_columns].abs()

    if values.isna().all():
        return "Unclassified"

    strongest_feature = values.idxmax()

    return FEATURE_LABELS.get(
        strongest_feature,
        "Unclassified"
    )


anomalies["dominant_weather_signal"] = anomalies.apply(
    identify_weather_signal,
    axis=1
)


# --------------------------------------------------
# 7. Assign prototype severity
# --------------------------------------------------

def assign_severity(score):

    ratio = score / gnn_threshold

    if ratio >= 3.0:
        return "Very High"

    elif ratio >= 2.0:
        return "High"

    elif ratio >= 1.5:
        return "Moderate"

    else:
        return "Low"


anomalies["severity"] = anomalies[
    "gnn_anomaly_score"
].apply(assign_severity)


# --------------------------------------------------
# 8. Group anomalies into time-bounded events
# --------------------------------------------------

print("\nGrouping anomalies into time-bounded events...")

records = anomalies.to_dict("records")

visited = set()
event_components = []

# Each event is anchored to its earliest detection.
# No event can span more than TIME_WINDOW_HOURS.
# This prevents unlimited chaining.

for start_index in range(len(records)):

    if start_index in visited:
        continue

    start_record = records[start_index]
    event_start_time = start_record["time"]

    queue = deque([start_index])
    visited.add(start_index)

    component = []

    while queue:

        current_index = queue.popleft()
        current = records[current_index]

        component.append(current)

        current_location = int(
            current["location_id"]
        )

        for other_index in range(len(records)):

            if other_index in visited:
                continue

            other = records[other_index]

            other_location = int(
                other["location_id"]
            )

            # Enforce the event's maximum time span.
            elapsed_hours = (
                other["time"] - event_start_time
            ).total_seconds() / 3600

            if elapsed_hours < 0:
                continue

            if elapsed_hours > TIME_WINDOW_HOURS:
                continue

            same_location = (
                current_location == other_location
            )

            graph_connected = (
                other_location
                in connected_locations[current_location]
            )

            if same_location or graph_connected:

                visited.add(other_index)
                queue.append(other_index)

    event_components.append(component)


# --------------------------------------------------
# 9. Create event summaries
# --------------------------------------------------

print("\nCreating improved event summaries...")

event_summaries = []
isolated_records = []

for event_number, component in enumerate(
    event_components,
    start=1
):

    event_df = pd.DataFrame(component)

    event_df["time"] = pd.to_datetime(
        event_df["time"]
    )

    start_time = event_df["time"].min()
    end_time = event_df["time"].max()

    duration_hours = (
        end_time - start_time
    ).total_seconds() / 3600

    location_ids = sorted(
        event_df["location_id"].unique()
    )

    location_names = [
        LOCATION_NAMES[int(location_id)]
        for location_id in location_ids
    ]

    dominant_signal = (
        event_df["dominant_weather_signal"]
        .value_counts()
        .idxmax()
    )

    maximum_score = event_df[
        "gnn_anomaly_score"
    ].max()

    severity = assign_severity(maximum_score)

    if len(event_df) == 1:
        event_category = "Isolated detection"
        isolated_records.append(component[0])
    else:
        event_category = "Multi-record candidate event"

    event_summaries.append({

        "event_id": f"EVT_{event_number:05d}",

        "start_time": start_time,

        "end_time": end_time,

        "duration_hours": round(
            duration_hours, 2
        ),

        "anomaly_records": len(event_df),

        "distinct_timestamps": event_df[
            "time"
        ].nunique(),

        "number_of_locations": len(location_ids),

        "location_ids": ",".join(
            map(str, location_ids)
        ),

        "locations": ", ".join(
            location_names
        ),

        "maximum_gnn_anomaly_score": maximum_score,

        "severity": severity,

        "dominant_weather_signal": dominant_signal,

        "event_category": event_category,

        "event_status": "Historical candidate",

        "event_type": "Unclassified weather anomaly"
    })


# --------------------------------------------------
# 10. Save event and isolated anomaly results
# --------------------------------------------------

event_results = pd.DataFrame(event_summaries)

event_results.to_csv(
    EVENT_FILE,
    index=False
)

isolated_df = pd.DataFrame(isolated_records)

if not isolated_df.empty:
    isolated_df.to_csv(
        ISOLATED_FILE,
        index=False
    )
else:
    pd.DataFrame().to_csv(
        ISOLATED_FILE,
        index=False
    )


# --------------------------------------------------
# 11. Print event statistics
# --------------------------------------------------

print("\n======================================")
print("IMPROVED EVENT TRACKING COMPLETED")
print("======================================")

print("Total anomaly records:", len(anomalies))

print("Total candidate groups:", len(event_results))

if not event_results.empty:

    print(
        "\nEvent category summary:"
    )

    print(
        event_results["event_category"].value_counts()
    )

    print(
        "\nSeverity summary:"
    )

    print(
        event_results["severity"].value_counts()
    )

    print(
        "\nDominant weather signal summary:"
    )

    print(
        event_results[
            "dominant_weather_signal"
        ].value_counts()
    )

    print(
        "\nFirst five candidate events:"
    )

    print(
        event_results.head().to_string(index=False)
    )

print("\nEvent file saved:")
print(EVENT_FILE)

print("\nIsolated anomaly file saved:")
print(ISOLATED_FILE)
