
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = [
    "datasets/processed/andhra_pradesh_weather_cleaned.csv",
    "datasets/processed/historical_features.csv",
    "datasets/raw/live_weather.csv",
    "datasets/raw/live_features.csv",

    "models/weather_scaler.pkl",
    "models/isolation_forest.pkl",
    "models/if_threshold.json",
    "models/gnn_autoencoder.pth",
    "models/gnn_threshold.json",
    "models/conditional_diffusion.pth",

    "outputs/weather_graph/weighted_adjacency_matrix.csv",

    "outputs/anomaly_results/historical_anomalies.csv",
    "outputs/anomaly_results/live_anomalies.csv",
    "outputs/anomaly_results/gnn_historical_anomalies.csv",
    "outputs/anomaly_results/model_comparison_test.csv",
    "outputs/anomaly_results/model_comparison_summary.json",
    "outputs/anomaly_results/tracked_weather_events.csv",
    "outputs/anomaly_results/isolated_weather_anomalies.csv",

    "outputs/diffusion/generated_weather.npy",
    "outputs/diffusion/generated_weather_comparison.png",
]


# ==================================================
# 2. VALIDATION HELPERS
# ==================================================

passed = 0
failed = 0


def report_check(name, condition, details=""):

    global passed, failed

    if condition:
        passed += 1
        print(f"[PASS] {name}")

    else:
        failed += 1
        print(f"[FAIL] {name}")

    if details:
        print(f"       {details}")


def get_csv(relative_path):

    path = PROJECT_ROOT / relative_path

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception as error:
        print(f"[ERROR] Could not read {relative_path}: {error}")
        return None


# ==================================================
# 3. CHECK REQUIRED FILES
# ==================================================

print("\n" + "=" * 60)
print("AI WEATHER ANOMALY PROJECT VALIDATION")
print("=" * 60)

print("\n1. CHECKING REQUIRED FILES")

for relative_path in REQUIRED_FILES:

    path = PROJECT_ROOT / relative_path

    report_check(
        f"File: {relative_path}",
        path.exists() and path.is_file()
    )


# ==================================================
# 4. CHECK HISTORICAL DATA
# ==================================================

print("\n2. CHECKING HISTORICAL DATA")

historical = get_csv(
    "datasets/processed/andhra_pradesh_weather_cleaned.csv"
)

if historical is not None:

    report_check(
        "Historical dataset is readable",
        not historical.empty,
        f"Rows: {len(historical)} | Columns: {len(historical.columns)}"
    )

    if "location_name" in historical.columns:

        location_count = historical["location_name"].nunique()

        report_check(
            "Historical dataset contains five locations",
            location_count == 5,
            f"Unique locations: {location_count}"
        )

    if "time" in historical.columns:

        times = pd.to_datetime(
            historical["time"],
            errors="coerce"
        )

        report_check(
            "Historical timestamps are valid",
            times.notna().all()
        )


# ==================================================
# 5. CHECK FEATURE DATA
# ==================================================

print("\n3. CHECKING MODEL FEATURES")

expected_features = [
    "temperature_2m_c",
    "relative_humidity_2m_pct",
    "precipitation_mm",
    "wind_speed_10m_kmh",
    "pressure_msl_hpa",
    "wind_gusts_10m_kmh",
    "wind_sin",
    "wind_cos"
]

historical_features = get_csv(
    "datasets/processed/historical_features.csv"
)

live_features = get_csv(
    "datasets/raw/live_features.csv"
)

for name, data in [
    ("Historical features", historical_features),
    ("Live features", live_features)
]:

    if data is None:
        report_check(f"{name} readable", False)
        continue

    missing = [
        column for column in expected_features
        if column not in data.columns
    ]

    report_check(
        f"{name} contain all eight model features",
        len(missing) == 0,
        f"Missing columns: {missing}"
    )

    if not missing:

        values = data[expected_features].to_numpy(
            dtype=float
        )

        report_check(
            f"{name} contain finite numeric values",
            np.isfinite(values).all()
        )

if live_features is not None:

    report_check(
        "Live features contain five locations",
        len(live_features) == 5,
        f"Rows: {len(live_features)}"
    )


# ==================================================
# 6. CHECK GRAPH
# ==================================================

print("\n4. CHECKING WEATHER GRAPH")

graph = get_csv(
    "outputs/weather_graph/weighted_adjacency_matrix.csv"
)

if graph is not None:

    graph_values = graph.iloc[:, 1:].to_numpy(
        dtype=float
    )

    report_check(
        "Graph adjacency matrix is 5 x 5",
        graph_values.shape == (5, 5),
        f"Shape: {graph_values.shape}"
    )

    if graph_values.shape == (5, 5):

        report_check(
            "Graph weights are finite",
            np.isfinite(graph_values).all()
        )

        report_check(
            "Graph is symmetric",
            np.allclose(
                graph_values,
                graph_values.T,
                atol=1e-6
            )
        )


# ==================================================
# 7. CHECK ANOMALY OUTPUTS
# ==================================================

print("\n5. CHECKING ANOMALY OUTPUTS")

gnn_events = get_csv(
    "outputs/anomaly_results/gnn_historical_anomalies.csv"
)

tracked_events = get_csv(
    "outputs/anomaly_results/tracked_weather_events.csv"
)

isolated_events = get_csv(
    "outputs/anomaly_results/isolated_weather_anomalies.csv"
)

if gnn_events is not None:

    report_check(
        "GNN historical anomaly output is readable",
        not gnn_events.empty,
        f"Rows: {len(gnn_events)}"
    )

if tracked_events is not None:

    required_event_columns = [
        "event_id",
        "start_time",
        "end_time",
        "anomaly_records",
        "distinct_timestamps",
        "number_of_locations",
        "locations",
        "maximum_gnn_anomaly_score",
        "severity",
        "dominant_weather_signal",
        "event_category"
    ]

    missing = [
        column for column in required_event_columns
        if column not in tracked_events.columns
    ]

    report_check(
        "Tracked events contain required columns",
        len(missing) == 0,
        f"Missing columns: {missing}"
    )

    report_check(
        "Tracked event IDs are unique",
        tracked_events["event_id"].is_unique
        if "event_id" in tracked_events.columns
        else False
    )

    report_check(
        "Tracked events contain 740 groups",
        len(tracked_events) == 740,
        f"Rows: {len(tracked_events)}"
    )

if isolated_events is not None:

    report_check(
        "Isolated anomaly output is readable",
        not isolated_events.empty,
        f"Rows: {len(isolated_events)}"
    )


# ==================================================
# 8. CHECK MODEL COMPARISON
# ==================================================

print("\n6. CHECKING MODEL COMPARISON")

comparison = get_csv(
    "outputs/anomaly_results/model_comparison_test.csv"
)

if comparison is not None:

    report_check(
        "Historical model comparison is readable",
        not comparison.empty,
        f"Rows: {len(comparison)}"
    )

summary_path = (
    PROJECT_ROOT
    / "outputs"
    / "anomaly_results"
    / "model_comparison_summary.json"
)

if summary_path.exists():

    try:

        with open(summary_path, "r") as file:
            summary = json.load(file)

        report_check(
            "Model comparison summary is valid JSON",
            isinstance(summary, dict)
        )

    except Exception as error:

        report_check(
            "Model comparison summary is valid JSON",
            False,
            str(error)
        )


# ==================================================
# 9. CHECK DIFFUSION OUTPUT
# ==================================================

print("\n7. CHECKING CONDITIONAL DIFFUSION")

diffusion_path = (
    PROJECT_ROOT
    / "outputs"
    / "diffusion"
    / "generated_weather.npy"
)

if diffusion_path.exists():

    try:

        generated = np.load(diffusion_path)

        report_check(
            "Generated weather field is readable",
            generated.size > 0,
            f"Shape: {generated.shape}"
        )

        report_check(
            "Generated weather values are finite",
            np.isfinite(generated).all()
        )

        report_check(
            "Generated values are within 0 to 1",
            generated.min() >= 0
            and generated.max() <= 1,
            f"Min: {generated.min():.6f} | "
            f"Max: {generated.max():.6f}"
        )

    except Exception as error:

        report_check(
            "Generated weather field is readable",
            False,
            str(error)
        )


# ==================================================
# 10. FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(f"Passed checks : {passed}")
print(f"Failed checks : {failed}")
print(f"Total checks  : {passed + failed}")

if failed == 0:

    print("\nSUCCESS: All executed validation checks passed.")

else:

    print("\nWARNING: Some checks need attention.")

print("\nNote:")
print(
    "These checks verify file availability, data structure "
    "and basic numerical integrity. They do not establish "
    "meteorological accuracy or model forecasting skill."
)

print("=" * 60)
