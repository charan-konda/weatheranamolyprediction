"""Diagnose ST-GNN reconstruction errors during global-city inference.

Run from the project root:
    python evaluate_stgnn_inference.py

This script reuses the same scaler, checkpoint, 24-hour sequence fetch,
feature order, and dynamic adjacency construction as backend/global_city_monitor.py.
It does not retrain or modify any model or dashboard file.

IMPORTANT: The ST-GNN was trained on five Andhra Pradesh coastal cities.
Comparisons with unfamiliar cities are diagnostic only. High reconstruction
error can reflect distribution shift and is not proof of extreme weather.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_FOLDER = PROJECT_ROOT / "backend"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "global_city_monitor" / "stgnn_diagnostics"

if not BACKEND_FOLDER.exists():
    raise FileNotFoundError(
        f"Backend folder not found: {BACKEND_FOLDER}\n"
        "Save this script in the project root beside the backend folder."
    )

if str(BACKEND_FOLDER) not in sys.path:
    sys.path.insert(0, str(BACKEND_FOLDER))

try:
    import global_city_monitor as monitor
except ImportError as error:
    raise ImportError(
        "Could not import backend.global_city_monitor. Confirm that the "
        "monitor engine is saved as backend/global_city_monitor.py and that "
        "backend/city_weather.py and backend/global_city_watchlist.py exist."
    ) from error


def get_city_key(location: dict) -> tuple[str, str]:
    return (
        str(location.get("city", "")).casefold(),
        str(location.get("country", "")).casefold(),
    )


def main() -> None:
    print("=" * 72)
    print("ST-GNN GLOBAL INFERENCE DIAGNOSTICS")
    print("=" * 72)
    print("This is a diagnostic comparison, not a weather-alert validation.\n")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    models = monitor.load_models()
    watchlist_cities = monitor.load_watchlist()
    if not watchlist_cities:
        raise RuntimeError("The global city watchlist is empty.")

    all_locations = monitor.REFERENCE_LOCATIONS + watchlist_cities
    sequence_by_city, common_times = monitor.prepare_all_sequences(watchlist_cities)

    # Match the global monitor's node order, feature order and scaler.
    node_sequences = []
    for location in all_locations:
        key = get_city_key(location)
        if key not in sequence_by_city:
            raise KeyError(f"No hourly sequence was fetched for {location.get('city')}.")

        rows = [monitor.validate_feature_row(row) for row in sequence_by_city[key]]
        scaled_rows = monitor.standardize_rows(rows, models["scaler"])
        node_sequences.append(scaled_rows)

    # Shape: [batch=1, 24 hours, graph nodes, 8 standardized features]
    graph_sequence = np.stack(node_sequences, axis=1)
    expected_shape = (
        monitor.SEQUENCE_LENGTH,
        len(all_locations),
        len(monitor.FEATURE_COLUMNS),
    )
    if graph_sequence.shape != expected_shape:
        raise ValueError(
            f"Unexpected input array shape {graph_sequence.shape}; expected {expected_shape}."
        )

    adjacency = monitor.build_dynamic_adjacency(
        watchlist_cities,
        models["original_adjacency"],
    )

    input_tensor = torch.tensor(
        graph_sequence[np.newaxis, ...], dtype=torch.float32
    )
    adjacency_tensor = torch.tensor(adjacency, dtype=torch.float32)

    model = models["stgnn_model"]
    model.eval()
    with torch.no_grad():
        reconstructed = model(input_tensor, adjacency_tensor)

    actual = input_tensor[0].cpu().numpy()          # [time, nodes, features]
    predicted = reconstructed[0].cpu().numpy()    # [time, nodes, features]

    if actual.shape != predicted.shape:
        raise ValueError("Model output shape does not match its input shape.")

    threshold = float(models["stgnn_threshold"])
    feature_names = list(monitor.FEATURE_COLUMNS)
    latest_timestamp = str(common_times[-1])

    city_rows = []
    feature_rows = []

    for node_index, location in enumerate(all_locations):
        latest_diff = actual[-1, node_index, :] - predicted[-1, node_index, :]
        window_diff = actual[:, node_index, :] - predicted[:, node_index, :]

        latest_feature_mse = np.square(latest_diff)
        latest_feature_mae = np.abs(latest_diff)
        window_feature_mse = np.mean(np.square(window_diff), axis=0)

        # This score matches the threshold definition used by the monitor:
        # mean squared reconstruction error across features at latest hour.
        latest_node_mse = float(np.mean(latest_feature_mse))
        window_node_mse = float(np.mean(np.square(window_diff)))
        reference_node = node_index < len(monitor.REFERENCE_LOCATIONS)

        city_name = location.get("city", "Unknown")
        country = location.get("country", "")
        city_rows.append({
            "city": city_name,
            "country": country,
            "group": "Training-region reference" if reference_node else "Global watchlist (unfamiliar inference)",
            "latest_hour_utc": latest_timestamp,
            "latest_node_mse": latest_node_mse,
            "training_threshold": threshold,
            "latest_threshold_exceeded": bool(latest_node_mse > threshold),
            "latest_error_to_threshold_ratio": latest_node_mse / threshold if threshold > 0 else np.nan,
            "24h_mean_node_mse": window_node_mse,
            "largest_latest_feature_error": feature_names[int(np.argmax(latest_feature_mse))],
        })

        for feature_index, feature_name in enumerate(feature_names):
            feature_rows.append({
                "city": city_name,
                "country": country,
                "group": "Training-region reference" if reference_node else "Global watchlist (unfamiliar inference)",
                "latest_hour_utc": latest_timestamp,
                "feature": feature_name,
                "latest_squared_error_standardized": float(latest_feature_mse[feature_index]),
                "latest_absolute_error_standardized": float(latest_feature_mae[feature_index]),
                "24h_mean_squared_error_standardized": float(window_feature_mse[feature_index]),
            })

    city_df = pd.DataFrame(city_rows)
    feature_df = pd.DataFrame(feature_rows)

    # Reference group is small (five cities), so show descriptive stats only.
    reference_scores = city_df.loc[
        city_df["group"] == "Training-region reference", "latest_node_mse"
    ]
    watchlist_scores = city_df.loc[
        city_df["group"] == "Global watchlist (unfamiliar inference)", "latest_node_mse"
    ]

    summary = {
        "diagnostic_run_time_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "latest_aligned_weather_time_utc": latest_timestamp,
        "sequence_length_hours": int(monitor.SEQUENCE_LENGTH),
        "graph_node_count": int(len(all_locations)),
        "watchlist_city_count": int(len(watchlist_cities)),
        "feature_count": int(len(feature_names)),
        "stgnn_training_threshold_latest_hour_node_mse": threshold,
        "reference_city_latest_mse_mean": float(reference_scores.mean()),
        "reference_city_latest_mse_median": float(reference_scores.median()),
        "watchlist_city_latest_mse_mean": float(watchlist_scores.mean()),
        "watchlist_city_latest_mse_median": float(watchlist_scores.median()),
        "reference_cities_above_threshold": int((reference_scores > threshold).sum()),
        "watchlist_cities_above_threshold": int((watchlist_scores > threshold).sum()),
        "interpretation_warning": (
            "High errors for unfamiliar cities may reflect distribution shift, "
            "dynamic-graph differences, or unusual weather. This diagnostic alone "
            "cannot distinguish these causes or confirm meteorological anomalies."
        ),
    }

    city_csv = OUTPUT_FOLDER / "stgnn_city_error_diagnostics.csv"
    feature_csv = OUTPUT_FOLDER / "stgnn_feature_error_diagnostics.csv"
    summary_json = OUTPUT_FOLDER / "stgnn_diagnostic_summary.json"

    city_df.sort_values("latest_node_mse", ascending=False).to_csv(city_csv, index=False)
    feature_df.sort_values(
        ["city", "latest_squared_error_standardized"],
        ascending=[True, False],
    ).to_csv(feature_csv, index=False)

    import json
    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\nDIAGNOSTIC SUMMARY")
    print("-" * 72)
    print(f"Latest aligned hourly timestamp (UTC): {latest_timestamp}")
    print(f"Graph nodes: {len(all_locations)} ({len(monitor.REFERENCE_LOCATIONS)} references + {len(watchlist_cities)} watchlist)")
    print(f"ST-GNN latest-hour threshold: {threshold:.9f}")
    print(
        "Reference cities above threshold: "
        f"{summary['reference_cities_above_threshold']}/{len(monitor.REFERENCE_LOCATIONS)}"
    )
    print(
        "Watchlist cities above threshold: "
        f"{summary['watchlist_cities_above_threshold']}/{len(watchlist_cities)}"
    )
    print(f"Reference mean latest error: {reference_scores.mean():.6f}")
    print(f"Watchlist mean latest error: {watchlist_scores.mean():.6f}")

    print("\nCITY-LEVEL RECONSTRUCTION ERRORS (highest first)")
    display_columns = [
        "city", "group", "latest_node_mse", "latest_error_to_threshold_ratio",
        "latest_threshold_exceeded", "largest_latest_feature_error",
    ]
    print(city_df.sort_values("latest_node_mse", ascending=False)[display_columns].to_string(
        index=False,
        formatters={
            "latest_node_mse": "{:.6f}".format,
            "latest_error_to_threshold_ratio": "{:.1f}x".format,
        },
    ))

    print("\nFILES SAVED")
    print(f"  City-level results:   {city_csv}")
    print(f"  Feature-level results:{feature_csv}")
    print(f"  Summary JSON:         {summary_json}")
    print("\nIMPORTANT:")
    print(summary["interpretation_warning"])
    print("\nST-GNN INFERENCE DIAGNOSTICS COMPLETED.")


if __name__ == "__main__":
    main()
