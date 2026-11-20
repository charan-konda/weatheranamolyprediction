
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# --------------------------------------------------
# 1. Project paths and settings
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "datasets/processed/historical_features.csv"
GRAPH_FILE = ROOT / "outputs/weather_graph/weighted_adjacency_matrix.csv"

IF_MODEL_FILE = ROOT / "models/isolation_forest.pkl"
IF_THRESHOLD_FILE = ROOT / "models/if_threshold.json"

GNN_MODEL_FILE = ROOT / "models/gnn_autoencoder.pth"
GNN_THRESHOLD_FILE = ROOT / "models/gnn_threshold.json"

OUTPUT_DIR = ROOT / "outputs/anomaly_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
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


# --------------------------------------------------
# 2. Define the same GNN architecture used in training
# --------------------------------------------------


class SpatialGraphAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.self_layer_1 = nn.Linear(input_dim, hidden_dim)
        self.neighbor_layer_1 = nn.Linear(input_dim, hidden_dim)

        self.self_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_layer_2 = nn.Linear(hidden_dim, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, adjacency):

        neighbors = torch.matmul(adjacency, x)

        h = torch.relu(
            self.self_layer_1(x) +
            self.neighbor_layer_1(neighbors)
        )

        neighbors_h = torch.matmul(adjacency, h)

        h = torch.relu(
            self.self_layer_2(h) +
            self.neighbor_layer_2(neighbors_h)
        )

        return self.decoder(h)



# --------------------------------------------------
# 3. Load historical data and arrange graph snapshots
# --------------------------------------------------

print("Loading historical features...")

df = pd.read_csv(FEATURE_FILE)
df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(["time", "location_id"]).reset_index(drop=True)

unique_times = np.sort(df["time"].unique())

print("Total timestamps:", len(unique_times))

# Use the same chronological 70/15/15 split.
train_end = int(len(unique_times) * 0.70)
validation_end = int(len(unique_times) * 0.85)

test_times = unique_times[validation_end:]

test_df = df[df["time"].isin(test_times)].copy()

print("Test timestamps:", len(test_times))
print("Test records:", len(test_df))


# --------------------------------------------------
# 4. Isolation Forest test predictions
# --------------------------------------------------

print("\nLoading Isolation Forest...")

if_model = joblib.load(IF_MODEL_FILE)

with open(IF_THRESHOLD_FILE, "r") as file:
    if_threshold_data = json.load(file)

if_threshold = float(
    if_threshold_data["threshold"]
)

# Isolation Forest decision scores:
# Lower scores indicate more anomalous records.

if_scores = if_model.decision_function(
    test_df[FEATURES].values
)

if_labels = if_scores <= if_threshold

test_df["if_score"] = if_scores
test_df["if_anomaly"] = if_labels


# --------------------------------------------------
# 5. Prepare the normalized adjacency matrix
# --------------------------------------------------

print("\nLoading graph adjacency matrix...")

adjacency = pd.read_csv(
    GRAPH_FILE,
    index_col=0
).values.astype(np.float32)

adjacency = adjacency + np.eye(len(LOCATION_IDS), dtype=np.float32)

degree = np.sum(adjacency, axis=1)

degree_inv_sqrt = np.power(degree, -0.5)

degree_matrix = np.diag(degree_inv_sqrt)

adjacency = (
    degree_matrix @ adjacency @ degree_matrix
)

adjacency_tensor = torch.tensor(
    adjacency,
    dtype=torch.float32
)


# --------------------------------------------------
# 6. Create test graph snapshots
# --------------------------------------------------

print("\nCreating test graph snapshots...")

snapshots = []

for timestamp in test_times:

    group = test_df[test_df["time"] == timestamp]

    group = group.sort_values("location_id")

    if group["location_id"].tolist() != LOCATION_IDS:
        raise ValueError(
            f"Incorrect or missing locations at {timestamp}"
        )

    snapshots.append(
        group[FEATURES].values.astype(np.float32)
    )

test_snapshots = np.stack(snapshots)

test_tensor = torch.tensor(
    test_snapshots,
    dtype=torch.float32
)


# --------------------------------------------------
# 7. Load trained GNN model
# --------------------------------------------------

print("\nLoading trained GNN...")

checkpoint = torch.load(
    GNN_MODEL_FILE,
    map_location="cpu",
    weights_only=True
)

# These are the settings used in the training script.
gnn_model = SpatialGraphAutoencoder(
    input_dim=len(FEATURES),
    hidden_dim=32
)

# Support either a saved state dictionary or a checkpoint.
if "model_state_dict" in checkpoint:
    gnn_model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    gnn_model.load_state_dict(checkpoint)

gnn_model.eval()

with open(GNN_THRESHOLD_FILE, "r") as file:
    gnn_threshold_data = json.load(file)

gnn_threshold = float(
    gnn_threshold_data["threshold"]
)


# --------------------------------------------------
# 8. Calculate GNN reconstruction anomaly scores
# --------------------------------------------------

print("\nCalculating GNN anomaly scores...")

with torch.no_grad():

    reconstructed = gnn_model(
        test_tensor,
        adjacency_tensor
    )

    reconstruction_errors = torch.mean(
        (test_tensor - reconstructed) ** 2,
        dim=2
    )

gnn_scores = reconstruction_errors.numpy().reshape(-1)

gnn_labels = gnn_scores > gnn_threshold

test_df["gnn_score"] = gnn_scores
test_df["gnn_anomaly"] = gnn_labels


# --------------------------------------------------
# 9. Compare the two models
# --------------------------------------------------

both = (
    test_df["if_anomaly"] &
    test_df["gnn_anomaly"]
)

if_only = (
    test_df["if_anomaly"] &
    ~test_df["gnn_anomaly"]
)

gnn_only = (
    ~test_df["if_anomaly"] &
    test_df["gnn_anomaly"]
)

neither = (
    ~test_df["if_anomaly"] &
    ~test_df["gnn_anomaly"]
)

total = len(test_df)

print("\n====================================")
print("MODEL COMPARISON ON TEST DATA")
print("====================================")

print("\nIsolation Forest:")
print("Anomalies:", int(test_df["if_anomaly"].sum()))
print(
    "Anomaly percentage:",
    round(test_df["if_anomaly"].mean() * 100, 2),
    "%"
)

print("\nGNN:")
print("Anomalies:", int(test_df["gnn_anomaly"].sum()))
print(
    "Anomaly percentage:",
    round(test_df["gnn_anomaly"].mean() * 100, 2),
    "%"
)

print("\nAgreement between models:")
print("Both detected:", int(both.sum()))
print("Only Isolation Forest:", int(if_only.sum()))
print("Only GNN:", int(gnn_only.sum()))
print("Both classified as normal:", int(neither.sum()))

print(
    "Agreement percentage:",
    round((both.sum() + neither.sum()) / total * 100, 2),
    "%"
)


# --------------------------------------------------
# 10. Save detailed comparison results
# --------------------------------------------------

comparison_columns = [
    "time",
    "location_id",
    "location_name",
    "if_score",
    "if_anomaly",
    "gnn_score",
    "gnn_anomaly"
]

comparison_file = (
    OUTPUT_DIR / "model_comparison_test.csv"
)

test_df[comparison_columns].to_csv(
    comparison_file,
    index=False
)

summary = {
    "test_records": int(total),
    "isolation_forest_anomalies": int(
        test_df["if_anomaly"].sum()
    ),
    "gnn_anomalies": int(
        test_df["gnn_anomaly"].sum()
    ),
    "both_detected": int(both.sum()),
    "isolation_forest_only": int(if_only.sum()),
    "gnn_only": int(gnn_only.sum()),
    "both_normal": int(neither.sum()),
    "agreement_percentage": round(
        (both.sum() + neither.sum()) / total * 100,
        2
    )
}

summary_file = (
    OUTPUT_DIR / "model_comparison_summary.json"
)

with open(summary_file, "w") as file:
    json.dump(summary, file, indent=4)

print("\nComparison completed successfully!")
print("Detailed results:", comparison_file)
print("Summary:", summary_file)
