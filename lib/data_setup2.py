from .data_utils import *
from torch.utils.data import DataLoader
from pathlib import Path
import random
import pandas as pd

# preparing data loaders
import torchio as tio

# Path for directories and csv files
# data directory
PDX_DIR = Path("/PATH/TO/BIDS_DIRECTOY")


def get_image_paths(
    participant_ids: list[str], labels: list[int]
) -> tuple[list[str], list[int]]:
    images = []
    targets = []
    cn_imgs, pd_imgs = 0, 0

    for p_id, label in zip(participant_ids, labels):
        for path in (PDX_DIR / f"sub-{p_id}").rglob("*reg.pt"):
            images.append(path.absolute())
            targets.append(label)
            if label == 0:
                cn_imgs += 1
            else:
                pd_imgs += 1

    print(f"PD images: {pd_imgs}, CN images: {cn_imgs}")
    return images, targets


def transforms(mri_tensor: torch.Tensor):
    transform_affine = tio.Compose(
        [
            tio.RandomAffine(
                degrees=10, scales=0, translation=5  # , translation=(-20, 20)
            ),
        ],
    )
    transform_bias = tio.Compose([tio.transforms.RandomBiasField(coefficients=0.1)])

    # affine transform
    out = transform_affine(mri_tensor)

    # bias field
    if random.random() > 0.5:
        out = transform_bias(out)

    # patch occlusion
    if random.random() > 0.5:
        x, y, z = (
            random.randint(20, 100),
            random.randint(20, 100),
            random.randint(20, 100),
        )
        d1, d2, d3 = random.randint(5, 10), random.randint(5, 10), random.randint(5, 10)

        out.squeeze()[
            x - d1 : x + d1,
            y - d2 : y + d2,
            z - d3 : z + d3,
        ] = 0

    return out


def transforms_affine(mri_tensor: torch.Tensor):
    transform_affine = tio.Compose(
        [
            tio.RandomAffine(
                degrees=10, scales=0, translation=5  # , translation=(-20, 20)
            ),
        ],
    )

    # affine transform
    out = transform_affine(mri_tensor)

    return out


def mri_transform(mri_tensor: torch.Tensor, label: float) -> torch.Tensor:
    """augmentation and transform for data

    Args:
        mri_tensor (torch.Tensor): source mri tensor
        label (float): source tensor class label
    Returns:
        torch.Tensor: augmented mri tensor
    """
    mri_tensor = nn.functional.interpolate(
        mri_tensor.unsqueeze(0), (128, 128, 128)
    ).squeeze(0)

    return mri_tensor


# list of all paths to training images as tensors
train_paths, val_paths, test_paths = [], [], []

# list of labels for each of the images in splits
train_labels, val_labels, test_labels = [], [], []

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
