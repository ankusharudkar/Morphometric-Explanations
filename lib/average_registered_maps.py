import torch
import torch.nn as nn
from data_utils import identity_deform_field_3D

# reg_vec function in this file was used to register 3D displacement maps to a reference space (MNI)

@torch.inference_mode
def to_forward_warp(disp, initial, iters=50, verbose=True):
    orig = initial
    idf = identity_deform_field_3D([128] * 3).permute(0, 4, 3, 2, 1).to(disp.device)

    pad = "reflection"
    corners = True
    # disp = torch.pad
    for i in range(iters):
        actual = nn.functional.grid_sample(
            disp,
            orig.permute(0, 2, 3, 4, 1),
            mode="bilinear",
            align_corners=corners,
            padding_mode=pad,
        )
        diff = actual - idf

        direction = torch.randn_like(diff)
        direction = direction / direction.norm(2, dim=1)
        scale = 5
        # adaptive step size
        d1 = diff.norm(2, dim=1) / scale
        d2 = torch.ones_like(d1).cuda() * 1 / 20
        d = torch.where(d1 < d2, d1, d2)
        diffi = direction * d

        # move towards
        towards = orig + diffi
        t1 = nn.functional.grid_sample(
            disp,
            towards.permute(0, 2, 3, 4, 1),
            mode="bilinear",
            align_corners=corners,
            padding_mode=pad,
        )
        diff1 = t1 - idf
        diff1 = diff1.norm(2, dim=1, keepdim=True)

        # move away
        away = orig - diffi
        t2 = nn.functional.grid_sample(
            disp,
            away.permute(0, 2, 3, 4, 1),
            mode="bilinear",
            align_corners=corners,
            padding_mode=pad,
        )
        diff2 = t2 - idf
        diff2 = diff2.norm(2, dim=1, keepdim=True)

        # pick direction that reduced distance
        orig1 = torch.where(diff1 < diff, towards, orig)
        orig = torch.where(diff1 < diff2, orig1, away)
        # orig = torch.where(diff1 < diff2, towards, away)

    if verbose:
        print(diff.abs().mean(), diff1.abs().mean(), diff2.abs().mean())

    return orig


@torch.inference_mode
def reg_vector_to_MNI(disp, idf, end_MNIB):
    start_pointsA = nn.functional.grid_sample(
        disp.cuda(),
        (end_MNIB).permute(0, 3, 2, 1, 4),
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )

    forward_MNI = to_forward_warp(
        end_MNIB.permute(0, 4, 3, 2, 1), idf.cuda(), 1000, False
    )

    start_pointMNIA = nn.functional.grid_sample(
        forward_MNI,
        start_pointsA.permute(0, 2, 3, 4, 1),
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )

    return start_pointMNIA


@torch.inference_mode
def reg_vec(t, d):
    idf = identity_deform_field_3D([128] * 3).permute(0, 4, 3, 2, 1).cuda()
    start_pointsA = nn.functional.grid_sample(
        t + idf,
        (d + idf).permute(0, 4, 3, 2, 1),
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    forward_MNI = to_forward_warp(d + idf, idf.cuda(), 1000, False)

    start_pointMNIA = nn.functional.grid_sample(
        forward_MNI,
        start_pointsA.permute(0, 4, 3, 2, 1),
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )

    return start_pointMNIA
