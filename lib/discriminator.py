import torch
from torch import nn


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        act = lambda: nn.LeakyReLU(0.1)

        block = lambda in_f, out_f: nn.Sequential(
            nn.Conv3d(in_f, out_f, 4, 2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm3d(out_f),
            act(),
        )

        self.layers = nn.Sequential(
            nn.Conv3d(2, 64, 4, 2, padding=1, padding_mode="reflect"),
            nn.InstanceNorm3d(64),
            act(),
            block(64, 128),
            block(128, 256),
            block(256, 512),
            nn.Conv3d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.InstanceNorm3d(512),
            act(),
            nn.Conv3d(512, 512, 3, 1, 1, padding_mode="reflect"),
        )

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.layers(x)
