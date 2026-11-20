
import numpy as np
import matplotlib.pyplot as plt

# Load simulated datasets
fine_fields = np.load("datasets/simulated/fine_weather.npy")
coarse_fields = np.load("datasets/simulated/coarse_weather.npy")

# Select the first sample
fine = fine_fields[0]
coarse = coarse_fields[0]

# Display both grids
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im1 = axes[0].imshow(
    coarse,
    cmap="Blues",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)
axes[0].set_title("Coarse Weather Grid (4x4)")
axes[0].set_xlabel("Grid X")
axes[0].set_ylabel("Grid Y")
fig.colorbar(im1, ax=axes[0], label="Simulated rainfall intensity")

im2 = axes[1].imshow(
    fine,
    cmap="Blues",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)
axes[1].set_title("Fine Weather Grid (16x16)")
axes[1].set_xlabel("Grid X")
axes[1].set_ylabel("Grid Y")
fig.colorbar(im2, ax=axes[1], label="Simulated rainfall intensity")

plt.suptitle("Synthetic Weather Field: Coarse vs Fine")
plt.tight_layout()

plt.savefig(
    "datasets/simulated/weather_grid_comparison.png",
    dpi=150
)

plt.show()

print("Visualization completed.")
print("Fine grid shape:", fine.shape)
print("Coarse grid shape:", coarse.shape)
