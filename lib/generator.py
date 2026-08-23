import torch
from torch import nn
import lightning as L
from .data_utils import identity_deform_field_3D


# Generator backbone
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.residual = residual
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm3d(out_channels),
            nn.Identity() if residual else nn.ReLU(),
        )

    def forward(self, x):
        if self.residual:
            return self.conv(x) + x
        else:
            return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[
            # 8,
            16,
            32,
            64,
            128,
            256,
            512,
        ],
        dropout=0.0,
    ):
        super().__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.dropout = dropout

        # downs
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # ups
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose3d(feature * 2, feature, 2, 2),
                    nn.InstanceNorm3d(feature),
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1] * 2),
        )

        self.final_conv = nn.Sequential(
            nn.Conv3d(
                features[0],
                features[0] // 2,
                1,
                1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                features[0] // 2,
                out_channels,
                1,
                1,
            ),
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def mult_init(model, scale=1):
    for name, param in model.named_parameters():
        param.data.mul_(scale)


class DiffeomorphicTransform(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.transform = nn.Sequential(
            UNet(
                in_channels,
                out_channels,
            )
        )
        mult_init(self.transform, 0.1)
        f = identity_deform_field_3D((128, 128, 128))
        self.register_buffer("f", f)

    def forward(self, x, y=None):
        # velocity field
        if y is None:
            v = self.transform(x)
        else:
            v = self.transform(torch.cat([x, y], dim=1))

        steps = 10
        # scaling and squaring integration
        transformed = v / 2**steps
        transformed = transformed
        for i in range(steps):
            transformed = transformed + nn.functional.grid_sample(
                transformed,
                (transformed + self.f.permute(0, 4, 3, 2, 1)).permute(0, 2, 3, 4, 1),
                mode="bilinear",
                align_corners=False,
            )

        out = nn.functional.grid_sample(
            x,
            (transformed + self.f.permute(0, 4, 3, 2, 1)).permute(0, 2, 3, 4, 1),
            mode="bilinear",
            align_corners=False,
        )

        return out, transformed + self.f.permute(0, 4, 3, 2, 1), transformed, v


class UNetUpsample(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[
            # 8,
            16,
            32,
            64,
            128,
            256,
            512,
        ],
        dropout=0.0,
    ):
        super().__init__()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.dropout = dropout

        # downs
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # ups
        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv3d(feature * 2, feature, 1, 1),
                    nn.InstanceNorm3d(feature),
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = nn.Sequential(
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1], residual=True),
            DoubleConv(features[-1], features[-1] * 2),
        )

        self.final_conv = nn.Sequential(
            nn.Conv3d(
                features[0],
                features[0] // 2,
                1,
                1,
            ),
            nn.ReLU(),
            nn.Conv3d(
                features[0] // 2,
                out_channels,
                1,
                1,
            ),
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class AdditiveTransform(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.transform = nn.Sequential(
            UNetUpsample(
                in_channels,
                out_channels,
            )
        )
        # mult_init(self.transform, 0.1)

    def forward(self, x, y=None):
        # velocity field
        if y is None:
            diff = self.transform(x)
        else:
            diff = self.transform(torch.cat([x, y], dim=1))

        return x + diff, diff


def smooth_loss_l2(y_pred, penalty="l2", loss_mult=1):
    dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

    if penalty == "l2":
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    grad = d / 3.0

    if loss_mult is not None:
        grad *= loss_mult
    return grad
