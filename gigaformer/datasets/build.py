import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import albumentations as A
import cv2
from torch.utils.data import Dataset

from .inaturalist import INatDataset
from .xview3 import XviewDataset

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


@dataclass
class TransformConfig:
    """General transform configuration"""

    names: Optional[List[str]] = field(
        default_factory=list
    )  # List of names of Albumentations transforms
    max_size: int = 2048  # For SmallestMaxSize
    height: int = 2048  # For RandomCrop
    width: int = 2048  # For RandomCrop
    probability: float = (
        0.5  # For all transforms TODO: This should be changed per transform
    )


@dataclass
class DataConfig:
    """General Dataset Configuration."""

    dataset: str = "xview3"
    """Name of the dataset to use."""
    dir: str = "/home/group/xview3"
    """Path to the dataset directory."""
    num_workers: int = 8
    """Number of workers for the data loader."""
    test: bool = False
    """Whether to use the test dataset."""
    crop_size: int = 512
    """Size of the crop for training."""
    val_crop_size: int = 512
    """Size of the crop for validation."""
    overlap: int = 10
    """Overlap of the crops for validation."""
    positive_ratio: float = 0.85
    """Ratio of positive samples in a batch."""
    fold: int = 77
    """Fold number."""
    folds_csv: str = "meta/folds.csv"
    """Path to csv for folds."""
    shoreline_dir: str = "/home/group/xview3/shoreline/validation"
    """Shoreline validation path."""
    multiplier: int = 64
    """Number of times to increase dataset by."""

    transforms: TransformConfig = field(default_factory=TransformConfig)


def create_data_datasets(config: DataConfig, test: bool = False):
    if (
        os.environ.get("RANK", "0") == "0"
    ):  # needed since distrbuted not initialized
        print("dataset config crop size", config.data.crop_size)
        if config.data.dataset == "xview3" and config.data.shoreline_dir:
            print("Legacy Warning:shoreline_dir is no longer used")
    if config.data.dataset == "xview3":
        if test:
            # TODO: Do we need to construct the Xview Dataset when testing?
            train_annotations = os.path.join(config.dir, "labels/public.csv")
            train_dataset = XviewDataset(
                mode="train",
                dataset_dir=config.dir,
                fold=12345,
                folds_csv=config.folds_csv,
                annotation_csv=train_annotations,
                crop_size=config.crop_size,
                multiplier=config.multiplier,
            )
            val_dataset = TestDataset(os.path.join(config.dir, "images/public"))
        else:
            train_annotations = os.path.join(
                config.dir, "labels/validation.csv"
            )
            train_dataset = XviewDataset(
                mode="train",
                dataset_dir=config.dir,
                fold=config.fold,
                folds_csv=config.folds_csv,
                annotation_csv=train_annotations,
                crop_size=config.crop_size,
                multiplier=config.multiplier,
                positive_ratio=config.positive_ratio,
            )
            val_dataset = XviewDataset(
                mode="val",
                dataset_dir=config.dir,
                fold=config.fold,
                folds_csv=config.folds_csv,
                annotation_csv=train_annotations,
                crop_size=config.val_crop_size,
            )
    elif config.data.dataset == "inaturalist":
        train_transforms = create_transforms(config.transforms, config.dataset)
        train_dataset = INatDataset(
            mode="train",
            dataset_dir=Path(config.data.dir),
            annotation_json="train2018.json",
            transforms=train_transforms,
        )
        val_dataset = INatDataset(
            mode="val",
            dataset_dir=Path(config.data.dir),
            annotation_json="val2018.json",
            transforms=create_transforms(config.transforms_val,config.dataset),
        )

    return train_dataset, val_dataset


def create_transforms(config: TransformConfig, dataset: str = "xview3"):
    transforms = []
    if dataset == "inaturalist":
        for transform in config.names:
            if transform == "RandomCrop":
                transforms.append(
                    A.RandomCrop(height=config.height, width=config.width)
                )
            elif transform == "SmallestMaxSize":
                transforms.append(A.SmallestMaxSize(max_size=config.max_size))
            elif transform == "CenterCrop":
                transforms.append(A.CenterCrop(height=config.height,width=config.width))
            elif transform == "Normalize":
                transforms.append(A.Normalize(mean=config.mean,std=config.std))

            else:
                raise NotImplementedError
    elif dataset == "xview3":
        return A.Compose(
            [
                # A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, p=0.3),
                #    A.HorizontalFlip(),
                #    A.VerticalFlip()
            ],
            additional_targets={
                "conf_mask": "mask",
                "length_mask": "mask",
                "vessel_mask": "mask",
                "fishing_mask": "mask",
                "center_mask": "mask",
            },
        )

    return A.Compose(transforms)


class TestDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.names = os.listdir(root_dir)

    def __getitem__(self, index):
        return dict(name=self.names[index])

    def __len__(self):
        return len(self.names)
