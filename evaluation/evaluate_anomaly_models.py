
from pathlib import Path
import json

import numpy as np
import pandas as pd


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "anomaly_results"
    / "model_comparison_test.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT
    / "outputs"
    / "anomaly_results"
)

SUMMARY_FILE = (
    OUTPUT_FOLDER
    / "anomaly_model_evaluation_summary.json"
)

LOCATION_FILE = (
    OUTPUT_FOLDER
    / "anomaly_model_evaluation_by_location.csv"
)

REQUIRED_COLUMNS = [
    "location_name",
    "time",
    "gnn_anomaly",
    "if_anomaly",
    "gnn_score",
    "if_score"
]


# ==================================================
# 2. LOAD TEST RESULTS
# ==================================================

print("\n" + "=" * 65)
print("GNN VS ISOLATION FOREST — HISTORICAL TEST EVALUATION")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Test comparison file not found: {INPUT_FILE}"
    )

data = pd.read_csv(INPUT_FILE)

if data.empty:
    raise ValueError("The comparison dataset is empty.")

missing_columns = [
    column for column in REQUIRED_COLUMNS
    if column not in data.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\n1. DATASET INFORMATION")
print(f"Test records: {len(data)}")
print(f"Locations: {data['location_name'].nunique()}")

data["time"] = pd.to_datetime(
    data["time"],
    errors="coerce"
)

if data["time"].isna().any():
    raise ValueError("Invalid timestamps found.")

# Convert boolean or numeric anomaly flags to 0 and 1.
for column in ["gnn_anomaly", "if_anomaly"]:

    normalized = data[column].map({
        True: 1,
        False: 0,
        1: 1,
        0: 0
    })

    if normalized.isna().any():
        raise ValueError(
            f"Unexpected values in {column}: "
            f"{data[column].unique()}"
        )

    data[column] = normalized.astype(int)

# Validate model scores.
for column in ["gnn_score", "if_score"]:

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

    if not np.isfinite(data[column].to_numpy()).all():
        raise ValueError(
            f"Invalid numerical values in {column}."
        )


# ==================================================
# 3. OVERALL DETECTION COUNTS
# ==================================================

print("\n2. OVERALL DETECTION COUNTS")

total = len(data)

gnn_count = int(data["gnn_anomaly"].sum())
if_count = int(data["if_anomaly"].sum())

both_anomaly = int(
    (
        (data["gnn_anomaly"] == 1)
        & (data["if_anomaly"] == 1)
    ).sum()
)

only_gnn = int(
    (
        (data["gnn_anomaly"] == 1)
        & (data["if_anomaly"] == 0)
    ).sum()
)

only_if = int(
    (
        (data["gnn_anomaly"] == 0)
        & (data["if_anomaly"] == 1)
    ).sum()
)

both_normal = int(
    (
        (data["gnn_anomaly"] == 0)
        & (data["if_anomaly"] == 0)
    ).sum()
)

agreement_count = both_anomaly + both_normal

agreement_percentage = (
    agreement_count / total
) * 100

disagreement_percentage = (
    (total - agreement_count) / total
) * 100

gnn_rate = (gnn_count / total) * 100
if_rate = (if_count / total) * 100

print(f"Total test records: {total}")
print(f"GNN detections: {gnn_count}")
print(f"Isolation Forest detections: {if_count}")
print(f"Both models detected anomaly: {both_anomaly}")
print(f"Only GNN detected anomaly: {only_gnn}")
print(f"Only Isolation Forest detected anomaly: {only_if}")
print(f"Both models classified normal: {both_normal}")

print(f"\nModel agreement: {agreement_percentage:.2f}%")
print(f"Model disagreement: {disagreement_percentage:.2f}%")

print(f"GNN detection rate: {gnn_rate:.2f}%")
print(f"Isolation Forest detection rate: {if_rate:.2f}%")


# ==================================================
# 4. SCORE DISTRIBUTIONS
# ==================================================

print("\n3. ANOMALY SCORE DISTRIBUTIONS")

score_summary = {}

for model_name, column in [
    ("GNN", "gnn_score"),
    ("Isolation Forest", "if_score")
]:

    quantiles = data[column].quantile(
        [0.50, 0.90, 0.95, 0.99]
    )

    values = {
        "minimum": float(data[column].min()),
        "maximum": float(data[column].max()),
        "mean": float(data[column].mean()),
        "median": float(data[column].median()),
        "std": float(data[column].std()),
        "p90": float(quantiles.loc[0.90]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99])
    }

    score_summary[model_name] = values

    print(f"\n{model_name}")

    for metric, value in values.items():
        print(f"{metric}: {value:.6f}")


# ==================================================
# 5. LOCATION-WISE COMPARISON
# ==================================================

print("\n4. LOCATION-WISE DETECTION BEHAVIOUR")

location_results = (
    data.groupby("location_name")
    .agg(
        test_records=("location_name", "size"),
        gnn_anomalies=("gnn_anomaly", "sum"),
        if_anomalies=("if_anomaly", "sum"),
        gnn_mean_score=("gnn_score", "mean"),
        if_mean_score=("if_score", "mean")
    )
    .reset_index()
)

location_results["gnn_detection_rate_pct"] = (
    location_results["gnn_anomalies"]
    / location_results["test_records"]
) * 100

location_results["if_detection_rate_pct"] = (
    location_results["if_anomalies"]
    / location_results["test_records"]
) * 100

location_results["anomaly_count_difference"] = (
    location_results["gnn_anomalies"]
    - location_results["if_anomalies"]
)

print(
    location_results.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)


# ==================================================
# 6. SAVE LOCATION RESULTS
# ==================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

location_results.to_csv(
    LOCATION_FILE,
    index=False
)


# ==================================================
# 7. SAVE OVERALL SUMMARY
# ==================================================

summary = {
    "evaluation_type": "Unsupervised model agreement analysis",
    "total_test_records": total,
    "number_of_locations": int(
        data["location_name"].nunique()
    ),
    "gnn_anomalies": gnn_count,
    "isolation_forest_anomalies": if_count,
    "both_models_detected_anomaly": both_anomaly,
    "only_gnn_detected_anomaly": only_gnn,
    "only_isolation_forest_detected_anomaly": only_if,
    "both_models_classified_normal": both_normal,
    "agreement_count": agreement_count,
    "agreement_percentage": agreement_percentage,
    "disagreement_percentage": disagreement_percentage,
    "gnn_detection_rate_percentage": gnn_rate,
    "isolation_forest_detection_rate_percentage": if_rate,
    "score_distributions": score_summary,
    "interpretation_limit": (
        "No independently verified ground-truth event labels "
        "are available. Agreement is not equivalent to accuracy."
    )
}

with open(SUMMARY_FILE, "w") as file:
    json.dump(summary, file, indent=4)


# ==================================================
# 8. FINAL REPORT
# ==================================================

print("\n" + "=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)

print(f"Test records: {total}")
print(f"GNN anomaly detections: {gnn_count}")
print(f"Isolation Forest detections: {if_count}")
print(f"Agreement: {agreement_percentage:.2f}%")

print("\nFiles saved:")
print(SUMMARY_FILE)
print(LOCATION_FILE)

print("\nIMPORTANT:")
print(
    "This analysis measures model agreement and detection "
    "behaviour. It does not establish classification accuracy, "
    "precision, recall, F1-score, or meteorological validity."
)

print("=" * 65)
