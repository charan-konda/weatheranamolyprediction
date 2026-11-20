
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------
# 1. PROJECT CONFIGURATION
# -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HISTORICAL_FILE = (
    PROJECT_ROOT / "datasets" / "processed"
    / "historical_features.csv"
)

LIVE_FILE = (
    PROJECT_ROOT / "datasets" / "raw"
    / "live_features.csv"
)

GRAPH_FILE = (
    PROJECT_ROOT / "outputs" / "weather_graph"
    / "weighted_adjacency_matrix.csv"
)

MODEL_FILE = PROJECT_ROOT / "models" / "gnn_autoencoder.pth"

THRESHOLD_FILE = (
    PROJECT_ROOT / "models" / "gnn_threshold.json"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT / "outputs" / "anomaly_results"
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

HIDDEN_DIM = 32
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001

THRESHOLD_PERCENTILE = 99

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cpu")


# -------------------------------------------------
# 2. GRAPH AUTOENCODER ARCHITECTURE
# -------------------------------------------------

class SpatialGraphAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        # First graph message-passing layer
        self.self_layer_1 = nn.Linear(
            input_dim, hidden_dim
        )

        self.neighbor_layer_1 = nn.Linear(
            input_dim, hidden_dim
        )

        # Second graph message-passing layer
        self.self_layer_2 = nn.Linear(
            hidden_dim, hidden_dim
        )

        self.neighbor_layer_2 = nn.Linear(
            hidden_dim, hidden_dim
        )

        # Reconstruct the original weather features
        self.decoder = nn.Linear(
            hidden_dim, input_dim
        )

        self.activation = nn.ReLU()

    def forward(self, x, adjacency):

        # x shape: [batch, locations, features]

        # Aggregate neighbouring node features
        neighbor_features = torch.matmul(
            adjacency, x
        )

        hidden = self.activation(
            self.self_layer_1(x)
            + self.neighbor_layer_1(neighbor_features)
        )

        # Second message-passing operation
        neighbor_hidden = torch.matmul(
            adjacency, hidden
        )

        hidden = self.activation(
            self.self_layer_2(hidden)
            + self.neighbor_layer_2(neighbor_hidden)
        )

        # Reconstruct input weather features
        reconstructed = self.decoder(hidden)

        return reconstructed


# -------------------------------------------------
# 3. LOAD DATA
# -------------------------------------------------

def load_data():

    print("Loading historical and live features...")

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
        ["location_id"]
    ).reset_index(drop=True)

    return historical_df, live_df


# -------------------------------------------------
# 4. BUILD GRAPH ADJACENCY
# -------------------------------------------------

def load_adjacency():

    print("Loading spatial graph...")

    adjacency_df = pd.read_csv(
        GRAPH_FILE,
        index_col=0
    )

    adjacency_df = adjacency_df.loc[
        LOCATION_NAMES, LOCATION_NAMES
    ]

    adjacency = adjacency_df.to_numpy(
        dtype=np.float32
    )

    # Add self-loops so every node also receives
    # its own weather information
    adjacency = (
        adjacency + np.eye(len(LOCATION_IDS))
    )

    # Symmetric degree normalization
    degrees = adjacency.sum(axis=1)

    degree_inverse_sqrt = np.diag(
        1.0 / np.sqrt(degrees)
    )

    normalized_adjacency = (
        degree_inverse_sqrt
        @ adjacency
        @ degree_inverse_sqrt
    )

    return torch.tensor(
        normalized_adjacency,
        dtype=torch.float32,
        device=DEVICE
    )


# -------------------------------------------------
# 5. CONVERT DATA INTO GRAPH SNAPSHOTS
# -------------------------------------------------

def create_graph_snapshots(df):

    print("Creating graph snapshots...")

    timestamps = np.sort(
        df["time"].unique()
    )

    snapshots = []

    for timestamp in timestamps:

        frame = df[
            df["time"] == timestamp
        ].sort_values("location_id")

        # Each timestamp must contain all five nodes
        if frame["location_id"].tolist() != LOCATION_IDS:
            raise ValueError(
                f"Missing or incorrectly ordered nodes at {timestamp}"
            )

        snapshot = frame[
            FEATURE_COLUMNS
        ].to_numpy(dtype=np.float32)

        snapshots.append(snapshot)

    return np.stack(snapshots), timestamps


# -------------------------------------------------
# 6. CHRONOLOGICAL DATA SPLIT
# -------------------------------------------------

def split_data(snapshots, timestamps):

    total = len(timestamps)

    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train_data = snapshots[:train_end]

    validation_data = snapshots[
        train_end:validation_end
    ]

    test_data = snapshots[
        validation_end:
    ]

    print("\nChronological graph split:")
    print("Training timestamps:", len(train_data))
    print("Validation timestamps:", len(validation_data))
    print("Test timestamps:", len(test_data))

    return (
        train_data,
        validation_data,
        test_data,
        train_end,
        validation_end
    )


# -------------------------------------------------
# 7. TRAIN GRAPH AUTOENCODER
# -------------------------------------------------

def train_model(train_data, validation_data, adjacency):

    print("\nInitializing GNN autoencoder...")

    model = SpatialGraphAutoencoder(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.MSELoss()

    train_tensor = torch.tensor(
        train_data,
        dtype=torch.float32
    )

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_tensor = torch.tensor(
        validation_data,
        dtype=torch.float32,
        device=DEVICE
    )

    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):

        model.train()

        total_train_loss = 0.0

        for (batch,) in train_loader:

            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            reconstructed = model(
                batch,
                adjacency
            )

            loss = loss_function(
                reconstructed,
                batch
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        # Validate after every epoch
        model.eval()

        with torch.no_grad():

            validation_reconstruction = model(
                validation_tensor,
                adjacency
            )

            validation_loss = loss_function(
                validation_reconstruction,
                validation_tensor
            ).item()

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Training loss: {average_train_loss:.6f} | "
            f"Validation loss: {validation_loss:.6f}"
        )

    # Restore best validation checkpoint
    model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": len(FEATURE_COLUMNS),
            "hidden_dim": HIDDEN_DIM,
            "feature_columns": FEATURE_COLUMNS,
            "location_ids": LOCATION_IDS
        },
        MODEL_FILE
    )

    print("\nBest GNN model saved:", MODEL_FILE)

    return model


# -------------------------------------------------
# 8. CALCULATE NODE-LEVEL ANOMALY SCORES
# -------------------------------------------------

def calculate_anomaly_scores(model, data, adjacency):

    model.eval()

    data_tensor = torch.tensor(
        data,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        reconstructed = model(
            data_tensor,
            adjacency
        )

        # Mean squared reconstruction error
        # for each timestamp and location
        errors = torch.mean(
            (data_tensor - reconstructed) ** 2,
            dim=2
        )

    return errors.cpu().numpy()


# -------------------------------------------------
# 9. MAIN EXECUTION
# -------------------------------------------------

def main():

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    historical_df, live_df = load_data()

    adjacency = load_adjacency()

    snapshots, timestamps = create_graph_snapshots(
        historical_df
    )

    (
        train_data,
        validation_data,
        test_data,
        train_end,
        validation_end
    ) = split_data(snapshots, timestamps)

    model = train_model(
        train_data,
        validation_data,
        adjacency
    )

    # Establish threshold using validation errors only
    validation_scores = calculate_anomaly_scores(
        model,
        validation_data,
        adjacency
    )

    threshold = float(
        np.percentile(
            validation_scores,
            THRESHOLD_PERCENTILE
        )
    )

    with open(THRESHOLD_FILE, "w") as file:

        json.dump(
            {
                "threshold": threshold,
                "validation_percentile": THRESHOLD_PERCENTILE,
                "score_type": "node_reconstruction_mse"
            },
            file,
            indent=4
        )

    print("\nGNN anomaly threshold:", threshold)

    # Score all historical snapshots for
    # retrospective analysis
    historical_scores = calculate_anomaly_scores(
        model,
        snapshots,
        adjacency
    )

    historical_results = historical_df.copy()

    historical_results["gnn_anomaly_score"] = (
        historical_scores.reshape(-1)
    )

    historical_results["gnn_anomaly"] = (
        historical_results["gnn_anomaly_score"]
        > threshold
    ).astype(int)

    historical_results["gnn_anomaly_label"] = (
        historical_results["gnn_anomaly"].map(
            {
                0: "Normal",
                1: "Anomaly"
            }
        )
    )

    historical_results.to_csv(
        OUTPUT_FOLDER / "gnn_historical_anomalies.csv",
        index=False
    )

    # Prepare live snapshot in location order
    live_snapshot = live_df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    if live_snapshot.shape != (
        len(LOCATION_IDS),
        len(FEATURE_COLUMNS)
    ):
        raise ValueError(
            "Live data must contain exactly five locations "
            "with eight features each."
        )

    live_scores = calculate_anomaly_scores(
        model,
        live_snapshot[np.newaxis, :, :],
        adjacency
    )[0]

    live_results = live_df.copy()

    live_results["gnn_anomaly_score"] = live_scores

    live_results["gnn_anomaly"] = (
        live_results["gnn_anomaly_score"] > threshold
    ).astype(int)

    live_results["gnn_anomaly_label"] = (
        live_results["gnn_anomaly"].map(
            {
                0: "Normal",
                1: "Anomaly"
            }
        )
    )

    live_results.to_csv(
        OUTPUT_FOLDER / "gnn_live_anomalies.csv",
        index=False
    )

    print("\nHistorical GNN anomaly summary:")

    print(
        historical_results[
            "gnn_anomaly_label"
        ].value_counts()
    )

    print("\nGNN anomalies by location:")

    print(
        historical_results.groupby(
            "location_name"
        )["gnn_anomaly"].sum()
    )

    print("\nLive GNN results:")

    print(
        live_results[
            [
                "location_name",
                "time",
                "gnn_anomaly_score",
                "gnn_anomaly_label"
            ]
        ].to_string(index=False)
    )

    print("\nGNN anomaly detection completed!")

    print("Results saved in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
