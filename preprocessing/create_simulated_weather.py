
import os
import numpy as np
import pandas as pd

# Reproducible simulated data
np.random.seed(42)

OUTPUT_DIR = "datasets/simulated"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of simulated weather samples
NUM_SAMPLES = 500

# Fine grid: 16 x 16
FINE_SIZE = 16

# Coarse grid: 4 x 4
COARSE_SIZE = 4

def create_weather_field():
    """Create one simulated spatial rainfall field."""

    x = np.linspace(-1, 1, FINE_SIZE)
    y = np.linspace(-1, 1, FINE_SIZE)
    xx, yy = np.meshgrid(x, y)

    # Randomize the location and intensity of a rainfall system
    center_x = np.random.uniform(-0.5, 0.5)
    center_y = np.random.uniform(-0.5, 0.5)

    intensity = np.random.uniform(0.4, 1.0)
    spread = np.random.uniform(0.15, 0.45)

    distance = (
        (xx - center_x) ** 2
        + (yy - center_y) ** 2
    )

    rainfall = intensity * np.exp(
        -distance / (2 * spread ** 2)
    )

    # Add small spatial noise
    noise = np.random.normal(
        0, 0.025, (FINE_SIZE, FINE_SIZE)
    )

    rainfall = rainfall + noise

    # Rainfall cannot be negative
    rainfall = np.clip(rainfall, 0, 1)

    return rainfall.astype(np.float32)


def create_coarse_field(fine_field):
    """Downsample a 16x16 field into a 4x4 field."""

    return fine_field.reshape(
        COARSE_SIZE, 4,
        COARSE_SIZE, 4
    ).mean(axis=(1, 3))


def main():

    coarse_fields = []
    fine_fields = []

    print("Generating simulated weather fields...")

    for i in range(NUM_SAMPLES):

        fine = create_weather_field()
        coarse = create_coarse_field(fine)

        fine_fields.append(fine)
        coarse_fields.append(coarse)

    fine_fields = np.array(fine_fields)
    coarse_fields = np.array(coarse_fields)

    np.save(
        os.path.join(OUTPUT_DIR, "fine_weather.npy"),
        fine_fields
    )

    np.save(
        os.path.join(OUTPUT_DIR, "coarse_weather.npy"),
        coarse_fields
    )

    print("\nSIMULATED DATASET CREATED")
    print("-------------------------")
    print("Fine grid shape:", fine_fields.shape)
    print("Coarse grid shape:", coarse_fields.shape)
    print("Number of samples:", NUM_SAMPLES)

    print("\nFiles saved:")
    print("datasets/simulated/fine_weather.npy")
    print("datasets/simulated/coarse_weather.npy")


if __name__ == "__main__":
    main()
