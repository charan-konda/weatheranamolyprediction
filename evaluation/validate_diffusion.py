
import numpy as np
from pathlib import Path

# -----------------------------------------
# 1. Load generated and coarse fields
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GENERATED_PATH = (
    PROJECT_ROOT / "outputs" / "diffusion" / "generated_weather.npy"
)

COARSE_PATH = (
    PROJECT_ROOT / "datasets" / "simulated" / "coarse_weather.npy"
)

generated = np.load(GENERATED_PATH)
coarse = np.load(COARSE_PATH)[0]

print("Starting weather-field plausibility checks...")

# -----------------------------------------
# 2. Numerical validity
# -----------------------------------------

finite_check = np.all(np.isfinite(generated))

print("\n1. Numerical validity:")
print("All values finite:", finite_check)

# -----------------------------------------
# 3. Range check
# -----------------------------------------

range_check = (
    np.min(generated) >= 0
    and np.max(generated) <= 1
)

print("\n2. Rainfall intensity range:")
print("Minimum:", float(np.min(generated)))
print("Maximum:", float(np.max(generated)))
print("Within 0-1 range:", range_check)

# -----------------------------------------
# 4. Coarse consistency
# -----------------------------------------

generated_coarse = generated.reshape(
    4, 4, 4, 4
).mean(axis=(1, 3))

coarse_error = np.mean(
    np.abs(generated_coarse - coarse)
)

print("\n3. Coarse consistency:")
print("Block-average MAE:", float(coarse_error))

# -----------------------------------------
# 5. Spatial smoothness
# -----------------------------------------

horizontal_difference = np.abs(
    np.diff(generated, axis=1)
)

vertical_difference = np.abs(
    np.diff(generated, axis=0)
)

smoothness = (
    horizontal_difference.mean()
    + vertical_difference.mean()
) / 2

print("\n4. Spatial smoothness:")
print("Mean neighbouring difference:", float(smoothness))

# -----------------------------------------
# 6. Overall diagnostic summary
# -----------------------------------------

print("\n====================================")
print(" WEATHER FIELD PLAUSIBILITY REPORT")
print("====================================")

print("Numerical validity:", "PASS" if finite_check else "FAIL")
print("Range check:", "PASS" if range_check else "FAIL")
print("Coarse consistency MAE:", f"{coarse_error:.6f}")
print("Spatial smoothness:", f"{smoothness:.6f}")

if finite_check and range_check:
    print("\nBasic numerical checks passed.")
else:
    print("\nSome numerical checks failed.")

print("\nValidation completed.")

# -----------------------------------------
# 7. Save report
# -----------------------------------------

output_dir = PROJECT_ROOT / "outputs" / "diffusion"
output_dir.mkdir(parents=True, exist_ok=True)

report_path = output_dir / "plausibility_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Weather Field Plausibility Report\n")
    file.write(f"Numerical validity: {finite_check}\n")
    file.write(f"Range check: {range_check}\n")
    file.write(f"Coarse consistency MAE: {coarse_error:.6f}\n")
    file.write(f"Spatial smoothness: {smoothness:.6f}\n")

print("Report saved:", report_path)
