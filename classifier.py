import torch
from torch import nn
import lightning as L
import torchmetrics
import gc
from lib.data_setup2 import *
from lib.classifier import *

train = TensorMRIDatasetLazy(
    train_paths, train_labels, lambda x, _: transforms(mri_transform(x, _))
)
val = TensorMRIDatasetLazy(val_paths, val_labels, mri_transform)
test = TensorMRIDatasetLazy(test_paths, test_labels, mri_transform)

# data loaders
BATCH_SIZE = 8
train_loader = DataLoader(train, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val, BATCH_SIZE)
test_loader = DataLoader(test, BATCH_SIZE)

from lightning.fabric.strategies.fsdp import FSDPStrategy

# logging utilities
from lightning.fabric.loggers import TensorBoardLogger
import os

# setting training evironment
logger = TensorBoardLogger(root_dir="./classifier", default_hp_metric=False)

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

# hyperparameters
LR = 1e-4
WEIGHT_DECAY = 1e-5

# instantiate modules
classifier = Classifier1(norm=lambda x: nn.BatchNorm3d(x))

opt_c = torch.optim.Adam(
    classifier.parameters(),
    LR,
    weight_decay=WEIGHT_DECAY,
)

classifier, opt_c = fabric.setup(classifier, opt_c)
train_loader = fabric.setup_dataloaders(train_loader)
val_loader = fabric.setup_dataloaders(val_loader)
test_loader = fabric.setup_dataloaders(test_loader)

classifier.train()

step = 0
step_ongoing = 0
for epoch in range(500):
    acc_train = [0, 0]
    acc_val = [0, 0]
    acc_test = [0, 0]
    train_loss = [0, 0]
    train_count = 0
    fabric.print(f"Epoch: {epoch}")

    # batch iteration
    classifier.train()
    for b_idx, batch in enumerate(train_loader):

        fabric.print(f"Batch: {b_idx}")
        opt_c.zero_grad()
        imgs, targets = batch

        y_hat, patch_hat = classifier(imgs.float())

        loss_1 = nn.functional.binary_cross_entropy_with_logits(
            y_hat.reshape(-1), targets.reshape(-1).float()
        )

        loss_2 = nn.functional.binary_cross_entropy_with_logits(
            patch_hat,
            fabric.to_device(
                torch.ones_like(patch_hat) * targets.reshape(-1, 1, 1, 1, 1)
            ),
        )

        loss = loss_1 + loss_2 + loss_1 * loss_2

        fabric.backward(loss)
        opt_c.step()

        train_loss[0] += loss_1.item()
        train_loss[1] += loss_2.item()
        train_count += 1

        acc_train[1] += imgs.shape[0]
        acc_train[0] += torch.sum((y_hat.detach().sigmoid() > 0.5) == targets).item()

        fabric.log_dict(
            {
                "loss_running": loss.detach().item(),
                "prob": y_hat[0].detach().item(),
            },
            step=step_ongoing,
        )

        step_ongoing += 1

        del loss, y_hat

    # validation
    classifier.eval()
    with torch.inference_mode():
        val_loss = 0
        val_count = 0
        for b_idx, batch in enumerate(val_loader):
            imgs, targets = batch
            y_hat = classifier(imgs.float())[0]

            loss = nn.functional.binary_cross_entropy_with_logits(
                y_hat.reshape(-1), targets.reshape(-1).float()
            )
            val_loss += loss.item()
            val_count += 1
            acc_val[1] += imgs.shape[0]
            acc_val[0] += torch.sum((y_hat.detach().sigmoid() > 0.5) == targets).item()

            del loss, y_hat

        test_loss = 0
        test_count = 0
        for b_idx, batch in enumerate(test_loader):
            imgs, targets = batch
            y_hat = classifier(imgs.float())[0]

            loss = nn.functional.binary_cross_entropy_with_logits(
                y_hat.reshape(-1), targets.reshape(-1).float()
            )
            test_loss += loss.item()
            test_count += 1
            acc_test[1] += imgs.shape[0]
            acc_test[0] += torch.sum((y_hat.detach().sigmoid() > 0.5) == targets).item()

            del loss, y_hat

    fabric.log_dict(
        {
            "train_loss": train_loss[0] / train_count,
            "patch_loss": train_loss[1] / train_count,
            "val_loss": val_loss / val_count,
            "test_loss": test_loss / test_count,
            "train_acc": acc_train[0] / acc_train[1],
            "val_acc": acc_val[0] / acc_val[1],
            "test_acc": acc_test[0] / acc_test[1],
        },
        step=step,
    )

    step += 1
    torch.cuda.empty_cache()
    gc.collect()

    torch.save(
        {
            "classifier": classifier.state_dict(),
            "desciption": {
                "architecture": "Fully convolutional layers CN",
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "loss": "BCE",
                "batch size": 8,
                "GPUs": 2,
            },
        },
        "./classifier.ckpt",
    )
