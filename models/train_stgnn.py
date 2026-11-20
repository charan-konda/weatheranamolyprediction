import json
import random
from pathlib import Path

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

GRAPH_FILE = (
    PROJECT_ROOT / "outputs" / "weather_graph"
    / "weighted_adjacency_matrix.csv"
)

MODEL_FILE = PROJECT_ROOT / "models" / "stgnn_autoencoder.pth"
THRESHOLD_FILE = PROJECT_ROOT / "models" / "stgnn_threshold.json"

OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "anomaly_results"
HISTORICAL_OUTPUT = OUTPUT_FOLDER / "stgnn_historical_anomalies.csv"

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

LOCATION_NAMES = [
    "Visakhapatnam",
    "Kakinada",
    "Machilipatnam",
    "Nellore",
    "Ongole",
]

# A 24-hour sequence lets the model learn daily temporal context.
SEQUENCE_LENGTH = 24
HIDDEN_DIM = 32
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
THRESHOLD_PERCENTILE = 99
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cpu")


# -------------------------------------------------
# 2. SPATIO-TEMPORAL GRAPH AUTOENCODER
# -------------------------------------------------

class SpatioTemporalGraphAutoencoder(nn.Module):
    """
    Spatial message passing is applied at every time step.
    A shared GRU then learns temporal patterns independently
    for each node. The decoder reconstructs every node's
    weather features at every time step.

    Input shape: [batch, sequence, nodes, features]
    Adjacency shape: [nodes, nodes]
    Output shape: [batch, sequence, nodes, features]
    """

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

        # Aggregate connected cities for every timestamp.
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

        # Run the same temporal model over each graph node.
        # Reshape [B,T,N,H] -> [B*N,T,H].
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


# -------------------------------------------------
# 3. LOAD HISTORICAL DATA AND GRAPH
# -------------------------------------------------

def load_historical_data():
    print("Loading historical weather features...")

    if not HISTORICAL_FILE.exists():
        raise FileNotFoundError(
            f"Historical feature file not found: {HISTORICAL_FILE}"
        )

    df = pd.read_csv(HISTORICAL_FILE, parse_dates=["time"])

    required_columns = (
        ["time", "location_name"] + FEATURE_COLUMNS
    )
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            "Historical feature file is missing columns: "
            + ", ".join(missing)
        )

    df = df[df["location_name"].isin(LOCATION_NAMES)].copy()
    df = df.sort_values(["time", "location_name"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No historical rows matched the five reference locations.")

    return df


def load_adjacency():
    print("Loading spatial graph...")

    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Weighted adjacency file not found: {GRAPH_FILE}"
        )

    adjacency_df = pd.read_csv(GRAPH_FILE, index_col=0)

    try:
        adjacency_df = adjacency_df.loc[LOCATION_NAMES, LOCATION_NAMES]
    except KeyError as error:
        raise ValueError(
            "The adjacency matrix does not contain all five expected locations."
        ) from error

    adjacency = adjacency_df.to_numpy(dtype=np.float32)

    if not np.isfinite(adjacency).all():
        raise ValueError("Adjacency matrix contains non-finite values.")

    # Add self-loops and apply symmetric degree normalization.
    adjacency = adjacency + np.eye(len(LOCATION_NAMES), dtype=np.float32)
    degrees = adjacency.sum(axis=1)

    if np.any(degrees <= 0):
        raise ValueError("Every graph node must have a positive degree.")

    degree_inverse_sqrt = np.diag(1.0 / np.sqrt(degrees))
    normalized = degree_inverse_sqrt @ adjacency @ degree_inverse_sqrt

    return torch.tensor(normalized, dtype=torch.float32, device=DEVICE)


# -------------------------------------------------
# 4. CREATE COMPLETE GRAPH SNAPSHOTS
# -------------------------------------------------

def create_graph_snapshots(df):
    print("Creating ordered graph snapshots...")

    timestamps = np.sort(df["time"].dropna().unique())
    snapshots = []

    for timestamp in timestamps:
        frame = df[df["time"] == timestamp].set_index("location_name")

        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate city rows found at {timestamp}.")

        if set(frame.index) != set(LOCATION_NAMES):
            # Do not silently create incomplete graph snapshots.
            continue

        frame = frame.loc[LOCATION_NAMES]

        values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite feature value found at timestamp {timestamp}."
            )

        snapshots.append(values)

    if len(snapshots) <= SEQUENCE_LENGTH * 3:
        raise ValueError(
            "Not enough complete hourly snapshots to create training, "
            "validation and test sequences."
        )

    return np.stack(snapshots), pd.to_datetime(timestamps[:len(snapshots)])


# -------------------------------------------------
# 5. CHRONOLOGICAL SEQUENCE SPLIT
# -------------------------------------------------

def make_sequences(data, sequence_length):
    sequences = []

    for end_index in range(sequence_length - 1, len(data)):
        start_index = end_index - sequence_length + 1
        sequences.append(data[start_index:end_index + 1])

    return np.stack(sequences)


def split_into_sequences(snapshots):
    total = len(snapshots)
    train_end = int(total * 0.70)
    validation_end = int(total * 0.85)

    train_snapshots = snapshots[:train_end]
    validation_snapshots = snapshots[train_end:validation_end]
    test_snapshots = snapshots[validation_end:]

    if min(
        len(train_snapshots),
        len(validation_snapshots),
        len(test_snapshots),
    ) < SEQUENCE_LENGTH:
        raise ValueError(
            "A chronological split is shorter than the required "
            f"{SEQUENCE_LENGTH}-hour sequence."
        )

    train_sequences = make_sequences(train_snapshots, SEQUENCE_LENGTH)
    validation_sequences = make_sequences(
        validation_snapshots, SEQUENCE_LENGTH
    )
    test_sequences = make_sequences(test_snapshots, SEQUENCE_LENGTH)

    print("\nChronological sequence split:")
    print("Training sequences:", len(train_sequences))
    print("Validation sequences:", len(validation_sequences))
    print("Test sequences:", len(test_sequences))
    print("Sequence length:", SEQUENCE_LENGTH, "hours")

    return (
        train_sequences,
        validation_sequences,
        test_sequences,
        train_end,
        validation_end,
    )


# -------------------------------------------------
# 6. TRAIN THE SPATIO-TEMPORAL GNN
# -------------------------------------------------

def train_model(train_sequences, validation_sequences, adjacency):
    print("\nInitializing Spatio-Temporal GNN autoencoder...")

    model = SpatioTemporalGraphAutoencoder(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dim=HIDDEN_DIM,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    loss_function = nn.MSELoss()

    train_tensor = torch.tensor(train_sequences, dtype=torch.float32)
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_tensor = torch.tensor(
        validation_sequences,
        dtype=torch.float32,
        device=DEVICE,
    )

    best_validation_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for (batch,) in train_loader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            reconstructed = model(batch, adjacency)
            loss = loss_function(reconstructed, batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            validation_reconstruction = model(
                validation_tensor,
                adjacency,
            )
            validation_loss = loss_function(
                validation_reconstruction,
                validation_tensor,
            ).item()

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Training loss: {average_train_loss:.6f} | "
            f"Validation loss: {validation_loss:.6f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "SpatioTemporalGraphAutoencoder",
            "input_dim": len(FEATURE_COLUMNS),
            "hidden_dim": HIDDEN_DIM,
            "feature_columns": FEATURE_COLUMNS,
            "reference_location_names": LOCATION_NAMES,
            "sequence_length": SEQUENCE_LENGTH,
            "best_validation_loss": best_validation_loss,
        },
        MODEL_FILE,
    )

    print("\nSpatio-Temporal GNN model saved:", MODEL_FILE)
    return model


# -------------------------------------------------
# 7. NODE-LEVEL SEQUENCE ANOMALY SCORES
# -------------------------------------------------

def calculate_sequence_scores(model, sequences, adjacency):
    model.eval()

    tensor = torch.tensor(
        sequences,
        dtype=torch.float32,
        device=DEVICE,
    )

    all_scores = []

    with torch.no_grad():
        for start in range(0, len(tensor), BATCH_SIZE):
            batch = tensor[start:start + BATCH_SIZE]
            reconstructed = model(batch, adjacency)

            # Score the final hour in each 24-hour input sequence.
            latest_actual = batch[:, -1, :, :]
            latest_reconstructed = reconstructed[:, -1, :, :]

            node_errors = torch.mean(
                (latest_actual - latest_reconstructed) ** 2,
                dim=2,
            )
            all_scores.append(node_errors.cpu().numpy())

    return np.concatenate(all_scores, axis=0)


def score_contiguous_period(model, snapshots, adjacency):
    sequences = make_sequences(snapshots, SEQUENCE_LENGTH)
    return calculate_sequence_scores(model, sequences, adjacency)


# -------------------------------------------------
# 8. MAIN EXECUTION
# -------------------------------------------------

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    df = load_historical_data()
    adjacency = load_adjacency()

    snapshots, timestamps = create_graph_snapshots(df)

    (
        train_sequences,
        validation_sequences,
        test_sequences,
        train_end,
        validation_end,
    ) = split_into_sequences(snapshots)

    model = train_model(
        train_sequences,
        validation_sequences,
        adjacency,
    )

    # Threshold is calculated only from validation-period scores.
    validation_scores = calculate_sequence_scores(
        model,
        validation_sequences,
        adjacency,
    )

    threshold = float(
        np.percentile(validation_scores, THRESHOLD_PERCENTILE)
    )

    with open(THRESHOLD_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "threshold": threshold,
                "validation_percentile": THRESHOLD_PERCENTILE,
                "score_type": "latest_hour_node_reconstruction_mse",
                "sequence_length_hours": SEQUENCE_LENGTH,
                "model_type": "SpatioTemporalGraphAutoencoder",
            },
            file,
            indent=4,
        )

    print("\nST-GNN anomaly threshold:", threshold)
    print("Test sequences available:", len(test_sequences))

    # Score each chronological partition independently so no
    # sequence crosses a train/validation/test boundary.
    train_scores = score_contiguous_period(
        model, snapshots[:train_end], adjacency
    )
    validation_scores_aligned = score_contiguous_period(
        model, snapshots[train_end:validation_end], adjacency
    )
    test_scores = score_contiguous_period(
        model, snapshots[validation_end:], adjacency
    )

    # Build a full-length array. The first 23 hours of each
    # partition cannot be scored without a complete 24-hour window.
    all_scores = np.full(
        (len(snapshots), len(LOCATION_NAMES)),
        np.nan,
        dtype=np.float32,
    )

    train_first_score = SEQUENCE_LENGTH - 1
    all_scores[train_first_score:train_end] = train_scores

    val_length = validation_end - train_end
    all_scores[
        train_end + SEQUENCE_LENGTH - 1:validation_end
    ] = validation_scores_aligned

    all_scores[
        validation_end + SEQUENCE_LENGTH - 1:
    ] = test_scores

    # Map snapshots back to the matching rows in the original table.
    snapshot_times = pd.Index(timestamps)
    output = df[df["time"].isin(snapshot_times)].copy()
    output = output.sort_values(["time", "location_name"]).reset_index(drop=True)

    time_to_index = {timestamp: index for index, timestamp in enumerate(snapshot_times)}
    node_to_index = {name: index for index, name in enumerate(LOCATION_NAMES)}

    output["stgnn_anomaly_score"] = [
        all_scores[
            time_to_index[pd.Timestamp(row.time)],
            node_to_index[row.location_name],
        ]
        for row in output.itertuples()
    ]

    output["stgnn_anomaly"] = (
        output["stgnn_anomaly_score"].notna()
        & (output["stgnn_anomaly_score"] > threshold)
    ).astype(int)

    output["stgnn_anomaly_label"] = output["stgnn_anomaly"].map(
        {0: "Normal", 1: "Anomaly"}
    )

    output.to_csv(HISTORICAL_OUTPUT, index=False)

    print("\nHistorical ST-GNN anomaly summary:")
    print(output["stgnn_anomaly_label"].value_counts())

    print("\nST-GNN anomalies by reference location:")
    print(
        output.groupby("location_name")["stgnn_anomaly"].sum()
    )

    print("\nST-GNN training completed.")
    print("Model:", MODEL_FILE)
    print("Threshold:", THRESHOLD_FILE)
    print("Historical results:", HISTORICAL_OUTPUT)
    print(
        "\nNote: This first ST-GNN checkpoint is trained on the "
        "existing five Andhra Pradesh locations. Its variable-node "
        "architecture can accept a dynamic city graph during inference, "
        "but anomaly results for unfamiliar cities are experimental "
        "until the model is trained and evaluated on broader regional "
        "and climatic datasets."
    )


if __name__ == "__main__":
    main()
