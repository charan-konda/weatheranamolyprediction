
import numpy as np
import pandas as pd
from pathlib import Path

# Project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Output folder
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "weather_graph"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Geographic coordinates of the five locations
LOCATIONS = [
    {
        "location_id": 0,
        "location_name": "Visakhapatnam",
        "latitude": 17.6868,
        "longitude": 83.2185
    },
    {
        "location_id": 1,
        "location_name": "Kakinada",
        "latitude": 16.9891,
        "longitude": 82.2475
    },
    {
        "location_id": 2,
        "location_name": "Machilipatnam",
        "latitude": 16.1809,
        "longitude": 81.1303
    },
    {
        "location_id": 3,
        "location_name": "Nellore",
        "latitude": 14.4426,
        "longitude": 79.9865
    },
    {
        "location_id": 4,
        "location_name": "Ongole",
        "latitude": 15.5057,
        "longitude": 80.0499
    }
]

# Number of nearest neighbours per node
K = 2


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two geographic coordinates."""

    earth_radius = 6371.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return earth_radius * c


def build_distance_matrix():

    number_of_locations = len(LOCATIONS)

    distances = np.zeros(
        (number_of_locations, number_of_locations)
    )

    for i in range(number_of_locations):

        for j in range(number_of_locations):

            distances[i, j] = calculate_distance(
                LOCATIONS[i]["latitude"],
                LOCATIONS[i]["longitude"],
                LOCATIONS[j]["latitude"],
                LOCATIONS[j]["longitude"]
            )

    return distances


def build_adjacency_matrix(distances):

    number_of_locations = len(LOCATIONS)

    adjacency = np.zeros(
        (number_of_locations, number_of_locations)
    )

    # Find the K nearest neighbours of each location
    for i in range(number_of_locations):

        nearest_nodes = np.argsort(
            distances[i]
        )[1:K + 1]

        for j in nearest_nodes:
            adjacency[i, j] = 1

    # Make graph undirected:
    # If A connects to B, B also connects to A
    adjacency = np.maximum(
        adjacency,
        adjacency.T
    )

    # Convert distances into similarity weights
    positive_distances = distances[distances > 0]

    sigma = np.mean(positive_distances)

    weights = np.exp(
        -(distances ** 2) / (2 * sigma ** 2)
    )

    # Keep weights only for connected nodes
    weighted_adjacency = adjacency * weights

    return adjacency, weighted_adjacency


def main():

    print("Building weather location graph...")

    distances = build_distance_matrix()

    adjacency, weighted_adjacency = (
        build_adjacency_matrix(distances)
    )

    # Create location labels
    location_names = [
        item["location_name"]
        for item in LOCATIONS
    ]

    # Save geographic distance matrix
    distance_df = pd.DataFrame(
        distances,
        index=location_names,
        columns=location_names
    )

    distance_df.to_csv(
        OUTPUT_FOLDER / "location_distances_km.csv"
    )

    # Save binary adjacency matrix
    adjacency_df = pd.DataFrame(
        adjacency,
        index=location_names,
        columns=location_names
    )

    adjacency_df.to_csv(
        OUTPUT_FOLDER / "adjacency_matrix.csv"
    )

    # Save weighted adjacency matrix
    weighted_df = pd.DataFrame(
        weighted_adjacency,
        index=location_names,
        columns=location_names
    )

    weighted_df.to_csv(
        OUTPUT_FOLDER / "weighted_adjacency_matrix.csv"
    )

    # Save graph edge list
    edge_indices = np.argwhere(adjacency == 1)

    np.save(
        OUTPUT_FOLDER / "edge_index.npy",
        edge_indices.T
    )

    edge_weights = weighted_adjacency[
        edge_indices[:, 0],
        edge_indices[:, 1]
    ]

    np.save(
        OUTPUT_FOLDER / "edge_weights.npy",
        edge_weights
    )

    print("\nGraph created successfully!")

    print("\nDistance matrix (km):")
    print(distance_df.round(2))

    print("\nAdjacency matrix:")
    print(adjacency_df)

    print("\nNumber of nodes:", len(LOCATIONS))
    print("Number of directed edges:", len(edge_indices))

    print("\nGraph files saved in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
