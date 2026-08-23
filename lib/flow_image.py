import torch
from torch import nn
import lightning as L
from .data_utils import identity_deform_field_3D

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
                    # nn.Upsample(scale_factor=2),
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


from lib.data_utils import identity_deform_field_3D


def mult_init(model, scale=1):
    for name, param in model.named_parameters():
        param.data.mul_(scale)


class DiffeomorphicTransform(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=3,
        features=[
            # 8,
            16,
            32,
            64,
            128,
            256,
            512,
        ],
        size=128,
    ):
        super().__init__()
        self.transform = nn.Sequential(
            UNet(
                in_channels,
                out_channels,
                features=features,
            )
        )
        mult_init(self.transform, 0.1)
        f = identity_deform_field_3D((size, size, size))
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


class HierarchicalTransform(nn.Module):
    def __init__(
        self,
        scales=[8, 4, 2, 1],
        backbone=lambda scale: DiffeomorphicTransform(
            4, 3, features=[32, 64, 128], size=128 // scale #3,.. for image diffusion
        ),
    ):
        super().__init__()

        # TODO validate scales to be divisible
        self.scales = scales
        self.networks = nn.ModuleList()
        for scale in scales:
            self.networks.append(backbone(scale))

    def forward(self, x, y):
        # step 1
        mov_img = nn.functional.interpolate(x, scale_factor=1 / 8, mode="trilinear")
        fix_img = nn.functional.interpolate(y, scale_factor=1 / 8, mode="trilinear")

        _, _, disp1, _ = self.networks[0](mov_img, fix_img)
        disp1 = nn.functional.interpolate(disp1, scale_factor=2, mode="trilinear")
        img_scaled = nn.functional.interpolate(x, scale_factor=1 / 4, mode="trilinear")
        out_img = nn.functional.grid_sample(
            img_scaled,
            (disp1 + self.networks[1].f.permute(0, 4, 3, 2, 1)).permute(0, 2, 3, 4, 1),
            mode="bilinear",
            align_corners=False,
        )

        # step 2
        fix_img = nn.functional.interpolate(y, scale_factor=1 / 4, mode="trilinear")

        _, _, disp2, _ = self.networks[1](out_img, fix_img)
        disp2 = nn.functional.interpolate(disp2, scale_factor=2, mode="trilinear")
        # residual disp connection
        disp2 = disp2 + nn.functional.interpolate(
            disp1, scale_factor=2, mode="trilinear"
        )

        img_scaled = nn.functional.interpolate(x, scale_factor=1 / 2, mode="trilinear")
        out_img = nn.functional.grid_sample(
            img_scaled,
            (disp2 + self.networks[2].f.permute(0, 4, 3, 2, 1)).permute(0, 2, 3, 4, 1),
            mode="bilinear",
            align_corners=False,
        )

        # step 3
        fix_img = nn.functional.interpolate(y, scale_factor=1 / 2, mode="trilinear")

        _, _, disp3, _ = self.networks[2](out_img, fix_img)
        disp3 = nn.functional.interpolate(disp3, scale_factor=2, mode="trilinear")
        # residual disp connection
        disp3 = disp3 + nn.functional.interpolate(
            disp2, scale_factor=2, mode="trilinear"
        )

        img_scaled = x
        out_img = nn.functional.grid_sample(
            img_scaled,
            (disp3 + self.networks[3].f.permute(0, 4, 3, 2, 1)).permute(0, 2, 3, 4, 1),
            mode="bilinear",
            align_corners=False,
        )

        # step 4
        fix_img = y

        _, _, disp4, _ = self.networks[3](out_img, fix_img)
        # residual disp connection
        disp4 = disp4 + disp3

        img_scaled = x
        out_img = nn.functional.grid_sample(
            img_scaled,
            (disp4 + self.networks[3].f.permute(0, 4, 3, 2, 1)).permute(0, 2, 3, 4, 1),
            mode="bilinear",
            align_corners=False,
        )

        return out_img, disp4

import math

class FlowImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = HierarchicalTransform()
        self.embedding_dim = 512

    def gen_t_embedding(self, t, max_positions=10000):
        t = t * max_positions
        half_dim = self.embedding_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        emb = emb.squeeze().view(t.shape[0], 1, 1, 4, self.embedding_dim//4).repeat(1, 1, 128, 128//4, 1)
        return emb.to(t.device)

    def forward(self, img_noisy, target, timestep):
        timestep = self.gen_t_embedding(timestep)

        out_img, disp = self.model(img_noisy, torch.cat([target, timestep], dim=1))

        return out_img, disp
    
class FlowDisp(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = HierarchicalTransform()
        self.embedding_dim = 512

    # def gen_t_embedding(self, t, max_positions=10000):
    #     t = t * max_positions
    #     half_dim = self.embedding_dim // 2
    #     emb = math.log(max_positions) / (half_dim - 1)
    #     emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
    #     emb = t[:, None] * emb[None, :]
    #     emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    #     emb = emb.squeeze().view(t.shape[0], 1, 1, 4, self.embedding_dim//4).repeat(1, 1, 128, 128//4, 1)
    #     return emb.to(t.device)
    
    def gen_t_embedding(self, t):
        emb = torch.ones(t.shape[0], 1, 128, 128, 128).to(t.device) * t[:, None, None, None, None]
        return emb

    def forward(self, img_noisy, target, timestep, real_img):
        timestep = self.gen_t_embedding(timestep)

        out_img, disp = self.model(torch.cat([img_noisy, target, timestep], dim=1), real_img)

        return out_img, disp

