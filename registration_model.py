import torch
from torch import nn
import lightning as L
import torchmetrics
import gc
from lib.data_setup2 import *
from lib.generator import *
from torchvision.utils import make_grid

train = TensorMRIDatasetLazy(
    train_paths, train_labels, lambda x, _: transforms_affine(mri_transform(x, _))
)
val = TensorMRIDatasetLazy(val_paths, val_labels, mri_transform)
test = TensorMRIDatasetLazy(test_paths, test_labels, mri_transform)

# data loaders
BATCH_SIZE = 4
train_loader = DataLoader(train, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, BATCH_SIZE)
test_loader = DataLoader(test, BATCH_SIZE)

# Load MNI template
import nilearn
import nilearn.datasets
from nilearn import plotting

MNI = nilearn.datasets.load_mni152_template()

image = torch.tensor(MNI.get_fdata())
immax, immin = torch.quantile(image.reshape(-1), 0.95), image.min()
image = (image - immin) / (immax - immin)

MNI = image.unsqueeze(0)

MNI = (
    nn.functional.interpolate(
        MNI.unsqueeze(0), size=(128, 128, 128), mode="trilinear", align_corners=False
    )
    .squeeze(0)
    .float()
)


# logging utilities
from lightning.fabric.strategies.fsdp import FSDPStrategy
from lightning.fabric.loggers import TensorBoardLogger
import os

# setting training evironment
logger = TensorBoardLogger(root_dir="./reg", default_hp_metric=False)

if "SC_NODES" in os.environ:
    print(f"Nodes: {os.environ["SC_NODES"]}")
    fabric = L.Fabric(
        strategy=FSDPStrategy(accelerator="gpu", sharding_strategy="NO_SHARD"),
        num_nodes=int(os.environ["SC_NODES"]),
        loggers=logger,
    )
else:
    fabric = L.Fabric(
        strategy=FSDPStrategy(accelerator="gpu", sharding_strategy="NO_SHARD"),
        loggers=logger,
    )
    # debug mode
    # fabric = L.Fabric(accelerator="cpu", loggers=logger)
fabric.launch()

# hyper-parameters
LR = 1e-4
WEIGHT_DECAY = 1e-5

# model
register = DiffeomorphicTransform(2, 3)
opt = torch.optim.Adam(
    register.parameters(),
    LR,
    weight_decay=WEIGHT_DECAY,
)

register, opt = fabric.setup(register, opt)
train_loader = fabric.setup_dataloaders(train_loader)
val_loader = fabric.setup_dataloaders(val_loader)

register.train()

step = 0
step_ongoing = 0

for epoch in range(500):
    train_count = 0
    mse_loss = 0
    fabric.print(f"Epoch: {epoch}")

    # batch iteration
    register.train()
    for b_idx, batch in enumerate(train_loader):
        fabric.print(f"Batch: {b_idx}")
        imgs, _ = batch

        if random.random() > 0.5:
            imgs2, _ = next(iter(train_loader))
            imgs2 = imgs2[: imgs.shape[0], :]
        else:
            imgs2 = fabric.to_device(
                torch.stack([transforms_affine(MNI) for _ in range(imgs.shape[0])])
            )

        imgs_reg, _, t, _ = register(imgs, imgs2)

        loss_reg = nn.functional.mse_loss(imgs_reg, imgs2)

        loss_smooth = smooth_loss_l2(t, "l1")

        loss = loss_reg + loss_smooth * 20 + loss_reg * loss_smooth

        fabric.backward(loss)
        opt.step()
        opt.zero_grad()

        fabric.log_dict(
            {
                "loss_running": loss_reg.detach().item(),
                "loss_smooth": loss_smooth.detach().item(),
            },
            step=step_ongoing,
        )

        step_ongoing += 1

        s = 50
        if b_idx % 2 == 0:
            image = make_grid(
                torch.cat(
                    [
                        (imgs)[:1, :, :, :, s],
                        (imgs2)[:1, :, :, :, s],
                        (imgs_reg)[:1, :, :, :, s],
                        (imgs)[:1, :, s, :, :],
                        (imgs2)[:1, :, s, :, :],
                        (imgs_reg)[:1, :, s, :, :],
                    ]
                ),
                nrow=3,
            )
            fabric.logger.experiment.add_image("Gen_images", image, step_ongoing)

            fabric.logger.experiment.add_image(
                "Move_images",
                make_grid(torch.cat([(t[0].squeeze() / t[0].max())[:, :, :, s]])),
                step_ongoing,
            )

    register.eval()
    with torch.inference_mode():
        val_loss = 0
        val_smooth = 0
        for b_idx, batch in enumerate(val_loader):
            fabric.print(f"Batch: {b_idx}")
            imgs, _ = batch

            if random.random() > 0.5:
                imgs2, _ = next(iter(train_loader))
                imgs2 = imgs2[: imgs.shape[0], :]
            else:
                imgs2 = fabric.to_device(
                    torch.stack([transforms_affine(MNI) for _ in range(imgs.shape[0])])
                )

            imgs_reg, _, t, _ = register(imgs, imgs2)

            val_loss += nn.functional.mse_loss(imgs_reg, imgs2)

            val_smooth += smooth_loss_l2(t)

        fabric.log_dict(
            {
                "val_loss": val_loss,
                "val_loss": val_smooth,
            },
            step=step,
        )

    step += 1
    torch.cuda.empty_cache()
    gc.collect()

    torch.save(
        {
            "registration": register.state_dict(),  # was classifier
            "desciption": {
                "architecture": "UNet Registration Network (Diffeomorphic(2,3))",
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "loss": "BCE",
                "batch size": 2,
                "GPUs": 2,
            },
        },
        "./registeration.ckpt",
    )
