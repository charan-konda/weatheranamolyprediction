
import torch
import torch.nn as nn
import torch.nn.functional as F

TIMESTEPS = 100


class ConditionalDiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, noisy_image, coarse_condition, timestep):
        coarse_upsampled = F.interpolate(
            coarse_condition,
            size=noisy_image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        timestep_channel = (
            timestep.float().view(-1, 1, 1, 1)
            / TIMESTEPS
        )

        timestep_channel = timestep_channel.expand(
            -1, 1,
            noisy_image.shape[-2],
            noisy_image.shape[-1]
        )

        model_input = torch.cat(
            [noisy_image, coarse_upsampled, timestep_channel],
            dim=1
        )

        return self.network(model_input)
