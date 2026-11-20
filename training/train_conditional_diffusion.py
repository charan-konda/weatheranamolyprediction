
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device("cpu")
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TIMESTEPS = 100

DATA_DIR = "datasets/simulated"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading simulated weather data...")

coarse = np.load(os.path.join(DATA_DIR, "coarse_weather.npy"))
fine = np.load(os.path.join(DATA_DIR, "fine_weather.npy"))

coarse = torch.tensor(coarse, dtype=torch.float32).unsqueeze(1)
fine = torch.tensor(fine, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(coarse, fine)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Coarse data shape:", coarse.shape)
print("Fine data shape:", fine.shape)

betas = torch.linspace(0.0001, 0.02, TIMESTEPS)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def extract(values, timesteps, shape):
    result = values[timesteps]
    return result.reshape(-1, *([1] * (len(shape) - 1)))


class ConditionalDiffusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, noisy_field, coarse_condition, timestep):

        condition = F.interpolate(
            coarse_condition,
            size=(16, 16),
            mode="bilinear",
            align_corners=False
        )

        time_channel = (
            timestep.float() / TIMESTEPS
        ).view(-1, 1, 1, 1).expand(-1, 1, 16, 16)

        model_input = torch.cat(
            [noisy_field, condition, time_channel],
            dim=1
        )

        return self.network(model_input)


model = ConditionalDiffusionModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\nConditional diffusion model created.")
print("\nStarting diffusion training...")

for epoch in range(EPOCHS):

    total_loss = 0.0

    for coarse_batch, fine_batch in loader:

        batch_size = fine_batch.shape[0]

        timestep = torch.randint(
            0, TIMESTEPS, (batch_size,)
        )

        noise = torch.randn_like(fine_batch)

        alpha_bar = extract(
            alpha_bars, timestep, fine_batch.shape
        )

        noisy_field = (
            torch.sqrt(alpha_bar) * fine_batch
            + torch.sqrt(1 - alpha_bar) * noise
        )

        predicted_noise = model(
            noisy_field, coarse_batch, timestep
        )

        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.6f}"
    )

model_path = os.path.join(
    MODEL_DIR, "conditional_diffusion.pth"
)

torch.save(model.state_dict(), model_path)

print("\n===================================")
print("DIFFUSION TRAINING COMPLETED")
print("===================================")
print("Model saved:", model_path)
print("Device used:", DEVICE)

