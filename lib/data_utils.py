#%% imports
import nibabel as nb
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
import random
import numpy as np
from scipy.stats import norm

#%% deformations
def deform_subvolume(image: torch.Tensor, disp: torch.Tensor, center: tuple) -> torch.Tensor:

    x, y, z = center
    _, h, w, d, _ = disp.shape

    image = image.unsqueeze(0)

    # subvolume to displace
    subvolume = image[
        :,
        :,
        abs(math.floor(x - h / 2)) : math.floor(x + h / 2),
        abs(math.floor(y - w / 2)) : math.floor(y + w / 2),
        abs(math.floor(z - d / 2)) : math.floor(z + d / 2),
    ]

    # apply defromation to subvolume
    subvolume_t = torch.nn.functional.grid_sample(
        subvolume.float(),
        disp.permute(0, 3, 2, 1, 4).float(),
        mode="bilinear",
        align_corners=True,
    )

    # displaced image
    img_t = image.clone()
    img_t[
        :,
        :,
        abs(math.floor(x - h / 2)) : math.floor(x + h / 2),
        abs(math.floor(y - w / 2)) : math.floor(y + w / 2),
        abs(math.floor(z - d / 2)) : math.floor(z + d / 2),
    ] = subvolume_t

    return img_t.squeeze(0)

# can be run once for all examples
def identity_deform_field_3D(shape: list = [200, 200, 200]):
    """Returns an identity transfromation grid for a given input tensor shape

    Args:
        shape (list, optional): Shape of the input tensor. Defaults to [200, 200, 200].

    Returns:
        nn.tensor: normalized displacement field for transform
    """
    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.type(torch.FloatTensor)

    # normalizing the coodinated between -1 and 1
    for i in range(3):
        grid[:, i, ...] = 2 * (grid[:, i, ...] / (shape[i] - 1) - 0.5)
    grid = grid.permute(0, 2, 3, 4, 1)

    return grid

def get_markov_disp(shape: tuple, field_func: callable):
    """Generates a markov displacement field for a give shape

    Args:
        shape (tuple): shape of displacement field
        field_func (callable): markov function taking x, y, z

    Returns:
        torch.tensor: output displacement field
    """
    disp_field = torch.zeros([1, *shape, 3])

    for i, j, k in np.ndindex(shape):
        disp_field[0,i,j,k, :] = field_func(i,j,k)

    return disp_field

def gaussian_force(x, y, z, push=True, strength=15, shape=(60,60,60)):
    """
    x,y,z is beteween (-1,-1,-1) to (1,1,1)
    """
    # magnitude: distance from center
    dist = ((x-shape[0]/2)**2 + (y-shape[1]/2)**2 + (z-shape[2]/2)**2)**0.5
    # prob of that dist
    prob = norm.pdf(dist, scale=strength) * strength
    
    # direction: outside, inline with x, y, z
    if push:
        x, y, z = prob, prob, prob 
    else:
        x, y, z = -prob, -prob, -prob

    return torch.tensor([x, y, z])

def gaussian_pinch(x, y, z, axes="z", push=True, strength=15, shape=(60,60,60), min_sigma=0.5):
    """
    x,y,z is beteween (-1,-1,-1) to (1,1,1)
    """
    direction = 1 if push else -1
    axes = axes.lower()
    # created discontinuity
    dist = ((x-shape[0]/2)**2 + (y-shape[1]/2)**2) ** 0.5
    t = max(1 - abs(z - shape[2]/2)/(shape[2]/2), 0.1) * strength
    # print(dist, strength)
    x = norm.pdf( dist, scale=t) * t
    y = norm.pdf( dist, scale=t) * t
    z = 0

    if "x" in axes:
        return torch.tensor([z, x, y]) * direction
    elif "y" in axes:
        return torch.tensor([x, z, y]) * direction
    else:
        return torch.tensor([x, y, z]) * direction
    
def smoothing_disp(mat: torch.tensor, band:float=1):
    _, x, y, z, _ = mat.shape
    disp = torch.zeros_like(mat)

    for i, j, k in np.ndindex(disp.shape[1:-1]):
        edge_dist = (((i-x/2)**2 + (j-y/2)**2 + (k-z/2)**2)**0.5)/(max(x,y,z)/2)

        if edge_dist >= 0.9:
            mult  =  0
        elif 0.9-band < edge_dist < 0.9:
            mult = (0.9 - edge_dist)/band
        else:
            mult = 1

        disp[0, i, j, k, :] = mult * mat[0, i, j, k, :] 

    return disp

#%% dataset split based on patient files
def split_patient_files(files: list[Path], split: list[int]) -> list[list[Path]]:
    """Splits nii files based on patient id

    Args:
        files (list[Path]): list of paths of directories sub-*
        split (list[int]): lenght of each split in dataset

    Returns:
        list[list[Path]]: split lists of nii files
    """
    samples = []
    random.shuffle(files)
    start_idx = 0

    for count in split:
        nii_files = []
        dir_files = files[start_idx:start_idx+count]
        for directory in dir_files:
            nii_files.extend(list(directory.rglob("*reg.pt")))
        samples.append(nii_files)
        start_idx += count

    return samples

#%% Data loader class
class TensorMRIDataset(Dataset):
    """
    Dataset of MRI tensors specified by file paths and labels with possible deformation and augmentations
    """

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        augmentation: callable = lambda x, label: x,
    ):
        self.paths = paths
        self.labels = labels

        self.data = []

        for i, (path, label) in enumerate(zip(paths, labels)):
            print(f"Loading {i+1}/{len(paths)}", end="\r")
            volume = torch.load(path, weights_only=False)
            volume = augmentation(volume, label)
            self.data.append(volume)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx): 
        return self.data[idx], self.labels[idx]
    

class TensorMRIDatasetLazy(Dataset):
    """
    Dataset of MRI tensors specified by file paths and labels with possible deformation and augmentations
    """

    def __init__(
        self,
        paths: list[Path],
        labels: list[int],
        augmentation: callable = lambda x, label: x,
    ):
        self.paths = paths
        self.labels = labels

        self.augmentation = augmentation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx): 
        volume = torch.load(self.paths[idx], weights_only=False)
        volume = self.augmentation(volume, self.labels[idx])

        return volume, self.labels[idx]
    


def simulate_disease_interactions(transforms=None):
    "With common feature - the classifier only paid attention to common - showing good generalization skills"
    idf = identity_deform_field_3D((40,40,40))
    disp = smoothing_disp(
        get_markov_disp(
            (40,40,40),
            lambda x, y, z: gaussian_pinch(x, y, z, "z", False, 5, (40,40,40)),
        ),
        0.2,
    ) + idf
    center = (60, 50, 40)


    disp2 = smoothing_disp(
        get_markov_disp(
            (40,40,40),
            lambda x, y, z: gaussian_force(x, y, z, True, 6, (40,40,40)),
        ),
        0.1,
    ) + idf
    center2 = (65, 80, 60)

    disp3 = smoothing_disp(
        get_markov_disp(
            (40,40,40),
            lambda x, y, z: gaussian_force(x, y, z, False, 6, (40,40,40)),
        ),
        0.2,
    ) + idf
    center3 = (50, 45, 25)

    def func(volume):
        if random.random() < 0.5:
            if transforms is not None:
                volume = transforms(volume)
            return volume, 0
        volume = deform_subvolume(volume, disp, center)
        if random.random() > 0.5:
            volume = deform_subvolume(volume, disp2, center2)
        else:
            volume = deform_subvolume(volume, disp3, center3)

        if transforms is not None:
            volume = transforms(volume)
        return volume, 1

    return func


def simulate_disease_interactions(transforms=None):
    idf = identity_deform_field_3D((40,40,40))
    disp2 = smoothing_disp(
        get_markov_disp(
            (40,40,40),
            lambda x, y, z: gaussian_force(x, y, z, True, 6, (40,40,40)),
        ),
        0.1,
    ) + idf
    center2 = (65, 80, 60)

    disp3 = smoothing_disp(
        get_markov_disp(
            (40,40,40),
            lambda x, y, z: gaussian_force(x, y, z, False, 6, (40,40,40)),
        ),
        0.1,
    ) + idf
    center3 = (50, 45, 25)

    def func(volume):
        if random.random() < 0.5:
            # if both present then not disease
            # if random.random() < 0.5:
            #     volume = deform_subvolume(volume, disp2, center2)
            #     volume = deform_subvolume(volume, disp3, center3)
            if transforms is not None:
                volume = transforms(volume)
            return volume, 0
        if random.random() > 0.5:
            volume = deform_subvolume(volume, disp2, center2)
        else:
            volume = deform_subvolume(volume, disp3, center3)

        if transforms is not None:
            volume = transforms(volume)
        return volume, 1

    return func

class TensorMRIDatasetLazySimulated(Dataset):
    """
    Dataset of MRI tensors specified by file paths and labels with possible deformation and augmentations
    """

    def __init__(
        self,
        paths: list[Path],
        augmentation: callable = lambda x, label: x,
        simulation: callable = lambda x: x,
    ):
        self.paths = paths

        self.augmentation = augmentation
        self.simulation = simulation

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx): 
        volume = torch.load(self.paths[idx], weights_only=False)
        label = 0
        
        volume = self.augmentation(volume, 0)
        
        volume, label = self.simulation(volume)

        return volume, label

    
# Example simulation function
# def simulate():
#     disp = smoothing_disp(
#         get_markov_disp(
#             (60, 60, 60),
#             lambda x, y, z: gaussian_force(x, y, z, True, 15, (60, 60, 60)),
#         ),
#         0.4,
#     ) + identity_deform_field_3D((60, 60, 60))
#     center = (111, 91, 90)

#     def func(volume, label):
#         if label == 1:
#             volume = deform_subvolume(volume, disp, center)
#         return volume

#     return func