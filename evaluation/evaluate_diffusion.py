
import numpy as np
from pathlib import Path

# -----------------------------------------
# 1. File paths
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GENERATED_PATH = (
    PROJECT_ROOT / "outputs" / "diffusion" / "generated_weather.npy"
)

FINE_PATH = (
    PROJECT_ROOT / "datasets" / "simulated" / "fine_weather.npy"
)

COARSE_PATH = (
    PROJECT_ROOT / "datasets" / "simulated" / "coarse_weather.npy"
)

# -----------------------------------------
# 2. Load weather fields
# -----------------------------------------

print("Loading weather fields...")

generated = np.load(GENERATED_PATH)
fine_data = np.load(FINE_PATH)
coarse_data = np.load(COARSE_PATH)

target = fine_data[0]
coarse = coarse_data[0]

print("Generated shape:", generated.shape)
print("Target shape:", target.shape)
print("Coarse shape:", coarse.shape)

# -----------------------------------------
# 3. Calculate MAE
# -----------------------------------------

mae = np.mean(np.abs(generated - target))

# -----------------------------------------
# 4. Calculate RMSE
# -----------------------------------------

rmse = np.sqrt(np.mean((generated - target) ** 2))

# -----------------------------------------
# 5. Calculate coarse consistency
# -----------------------------------------

# Average each 4x4 block of the generated 16x16 field
generated_coarse = generated.reshape(
    4, 4, 4, 4
).mean(axis=(1, 3))

coarse_mae = np.mean(
    np.abs(generated_coarse - coarse)
)

# -----------------------------------------
# 6. Display results
# -----------------------------------------

print("\n====================================")
print(" CONDITIONAL DIFFUSION EVALUATION")
print("====================================")

print(f"MAE: {mae:.6f}")
print(f"RMSE: {rmse:.6f}")

print(
    f"Coarse consistency MAE: {coarse_mae:.6f}"
)

print("\nEvaluation completed.")

# -----------------------------------------
# 7. Save evaluation results
# -----------------------------------------

output_dir = PROJECT_ROOT / "outputs" / "diffusion"
output_dir.mkdir(parents=True, exist_ok=True)

results_path = output_dir / "evaluation_results.txt"

with open(results_path, "w", encoding="utf-8") as file:
    file.write("Conditional Diffusion Evaluation\n")
    file.write(f"MAE: {mae:.6f}\n")
    file.write(f"RMSE: {rmse:.6f}\n")
    file.write(
        f"Coarse consistency MAE: {coarse_mae:.6f}\n"
    )

print("Results saved:", results_path)
