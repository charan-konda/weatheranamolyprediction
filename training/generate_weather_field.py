
import os
import sys
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATH CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.conditional_diffusion_model import (
    ConditionalDiffusionModel,
    TIMESTEPS
)

# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

DEVICE = torch.device("cpu")

MODEL_PATH = PROJECT_ROOT / "models" / "conditional_diffusion.pth"

COARSE_PATH = (
    PROJECT_ROOT / "datasets" / "simulated" / "coarse_weather.npy"
)

FINE_PATH = (
    PROJECT_ROOT / "datasets" / "simulated" / "fine_weather.npy"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "diffusion"


# --------------------------------------------------
# 3. MAIN GENERATION FUNCTION
# --------------------------------------------------

def main():

    print("======================================")
    print(" CONDITIONAL WEATHER FIELD GENERATION")
    print("======================================")

    # Check required files
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not COARSE_PATH.exists():
        raise FileNotFoundError(
            f"Coarse weather data not found: {COARSE_PATH}"
        )

    if not FINE_PATH.exists():
        raise FileNotFoundError(
            f"Fine weather data not found: {FINE_PATH}"
        )

    # --------------------------------------------------
    # 4. LOAD TRAINED DIFFUSION MODEL
    # --------------------------------------------------

    print("\nLoading trained diffusion model...")

    model = ConditionalDiffusionModel().to(DEVICE)

    model_weights = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(model_weights)

    model.eval()

    print("Model weights loaded successfully.")

    # --------------------------------------------------
    # 5. LOAD SIMULATED WEATHER DATA
    # --------------------------------------------------

    print("\nLoading simulated weather data...")

    coarse_data = np.load(COARSE_PATH)
    fine_data = np.load(FINE_PATH)

    print("Coarse dataset shape:", coarse_data.shape)
    print("Fine dataset shape:", fine_data.shape)

    # Use the first sample for demonstration
    coarse = torch.tensor(
        coarse_data[0],
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    target = fine_data[0]

    print("Selected coarse field:", tuple(coarse.shape))
    print("Selected fine target:", target.shape)

    # --------------------------------------------------
    # 6. CREATE DIFFUSION NOISE SCHEDULE
    # --------------------------------------------------

    print("\nPreparing reverse diffusion schedule...")

    betas = torch.linspace(
        0.0001,
        0.02,
        TIMESTEPS,
        device=DEVICE
    )

    alphas = 1.0 - betas

    alpha_bars = torch.cumprod(
        alphas,
        dim=0
    )

    # --------------------------------------------------
    # 7. START REVERSE DIFFUSION
    # --------------------------------------------------

    print("\nStarting reverse diffusion sampling...")

    # Start with random noise
    generated = torch.randn(
        (1, 1, 16, 16),
        device=DEVICE
    )

    with torch.no_grad():

        for t in reversed(range(TIMESTEPS)):

            timestep = torch.tensor(
                [t],
                dtype=torch.long,
                device=DEVICE
            )

            # Predict noise
            predicted_noise = model(
                generated,
                coarse,
                timestep
            )

            beta_t = betas[t]
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]

            # Calculate reverse diffusion mean
            mean = (
                generated
                - (
                    beta_t
                    / torch.sqrt(1.0 - alpha_bar_t)
                ) * predicted_noise
            ) / torch.sqrt(alpha_t)

            # Add sampling noise except at final step
            if t > 0:

                alpha_bar_previous = alpha_bars[t - 1]

                posterior_variance = (
                    beta_t
                    * (1.0 - alpha_bar_previous)
                    / (1.0 - alpha_bar_t)
                )

                noise = torch.randn_like(generated)

                generated = (
                    mean
                    + torch.sqrt(posterior_variance) * noise
                )

            else:

                generated = mean

            if t % 20 == 0:
                print(f"Reverse diffusion timestep: {t}")

    # --------------------------------------------------
    # 8. PROCESS GENERATED WEATHER FIELD
    # --------------------------------------------------

    generated = torch.clamp(
        generated,
        min=0.0,
        max=1.0
    )

    generated_field = (
        generated.squeeze().cpu().numpy()
    )

    # --------------------------------------------------
    # 9. SAVE GENERATED WEATHER FIELD
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    generated_path = (
        OUTPUT_DIR / "generated_weather.npy"
    )

    np.save(
        generated_path,
        generated_field
    )

    print("\nGenerated weather field saved.")

    # --------------------------------------------------
    # 10. VISUALIZE RESULTS
    # --------------------------------------------------

    print("\nCreating comparison visualization...")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5)
    )

    axes[0].imshow(
        coarse_data[0],
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    axes[0].set_title(
        "Coarse Input (4x4)"
    )

    axes[1].imshow(
        target,
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    axes[1].set_title(
        "Original Fine Field (16x16)"
    )

    axes[2].imshow(
        generated_field,
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    axes[2].set_title(
        "Generated Fine Field (16x16)"
    )

    for ax in axes:
        ax.set_xlabel("Grid Column")
        ax.set_ylabel("Grid Row")

    plt.tight_layout()

    comparison_path = (
        OUTPUT_DIR / "generated_weather_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    # --------------------------------------------------
    # 11. BASIC NUMERICAL DIAGNOSTICS
    # --------------------------------------------------

    print("\n======================================")
    print(" WEATHER FIELD GENERATION COMPLETED")
    print("======================================")

    print("Generated field shape:", generated_field.shape)

    print(
        "Generated minimum:",
        float(generated_field.min())
    )

    print(
        "Generated maximum:",
        float(generated_field.max())
    )

    print(
        "Generated mean:",
        float(generated_field.mean())
    )

    print("Saved generated field:", generated_path)

    print("Saved comparison image:", comparison_path)


# --------------------------------------------------
# 12. PROGRAM ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()
