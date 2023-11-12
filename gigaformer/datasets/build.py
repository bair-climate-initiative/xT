#### Lot of code taken from InternImage/Swin
#### https://github.com/OpenGVLab/InternImage/blob/3e083be9c807793ec1d6a9ffe091978ee01de02b/classification

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import albumentations as A
import cv2
import torch
from timm.data import Mixup, create_transform
from torch.utils.data import Dataset
from torchvision import transforms

from ..utils import get_rank, get_world_size
from .inaturalist import INatDataset
from .xview3 import XviewDataset

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


try:
    from torchvision.transforms import InterpolationMode

    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            return InterpolationMode.BILINEAR
except:
    from timm.data.transforms import _pil_interp


@dataclass
class AugmentationConfig:
    """Configuration for augmentation related parameters."""

    auto_augment: str = "rand-m9-mstd0.5-inc1"
    """AutoAugment shorthand (default: 'rand-m9-mstd0.5-inc1')"""
    color_jitter: float = 0.4
    """Color jitter factor (default: 0.4)"""
    reprob: float = 0.0
    """Random erase prob (default: 0.)"""
    remode: str = "pixel"
    """Random erase mode (default: 'const')"""
    recount: int = 1
    """Random erase count (default: 1)"""
    mixup: float = 0.0
    """Mixup alpha, mixup enabled if > 0"""
    cutmix: float = 0.0
    """Cutmix alpha, cutmix enabled if > 0"""
    cutmix_minmax: Any = None
    """Cutmix min/max ratio, overrides alpha and enables cutmix if set"""
    mixup_prob: float = 1.0
    """Probability of performing mixup or cutmix when either/both is enabled"""
    mixup_switch_prob: float = 0.5
    """Probability of switching to cutmix when both mixup and cutmix enabled"""
    mixup_mode: str = 'batch'
    """How to apply mixup/cutmix params. Per batch, pair, or elem."""
    label_smoothing: float = 0.0
    """Label smoothing factor."""

    random_resized_crop: bool = False
    """Whether to use random resized crop for testing. config.data.test_crop takes priority."""
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    """Mean of the dataset."""
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    """Standard deviation of the dataset."""


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
    batch_size: int = 4 
    """Batch size for training."""
    val_batch_size: int = 1
    """Batch size for validation."""
    crop_size: int = 512
    """Size of the crop for training."""
    num_classes: int = 8142 
    """Number of classes in the dataset."""
    test_crop: bool = True
    """Whether to use a center crop for testing."""
    interpolation: str = "bilinear"
    """Interpolation method for resizing."""

    """Xview3 pertinent settings."""
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

    """iNaturalist pertinent settings."""
    supercategories: Optional[List[str]] = field(default_factory=list)
    """iNaturalist only, the list of supercategories to filter by"""
    category_label_path: str = "meta/category_label_reptilia_map.json"
    """iNaturalist only, path to category label map if provided."""

    aug: AugmentationConfig = field(default_factory=AugmentationConfig)


def build_loader(config: DataConfig, test: bool = False):
    if (
        os.environ.get("RANK", "0") == "0"
    ):  # needed since distrbuted not initialized
        print("dataset config crop size", config.crop_size)
        if config.dataset == "xview3" and config.shoreline_dir:
            print("Legacy Warning:shoreline_dir is no longer used")
    if config.dataset == "xview3":
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
    elif config.dataset == "inaturalist":
        train_transforms = create_imagenet_transforms(config, is_train=True)
        train_dataset = INatDataset(
            mode="train",
            dataset_dir=Path(config.dir),
            annotation_json="train2018.json",
            categories_json="categories.json",
            supercategories=config.supercategories,
            category_label_path=config.category_label_path,
            channels_first=True,
            transforms=train_transforms,
        )
        val_transforms = create_imagenet_transforms(config, is_train=False)
        val_dataset = INatDataset(
            mode="val",
            dataset_dir=Path(config.dir),
            annotation_json="val2018.json",
            categories_json="categories.json",
            supercategories=config.supercategories,
            category_label_path=config.category_label_path,
            channels_first=True,
            transforms=val_transforms
        )
    
    print(f"Rank {get_rank()} saw world size {get_world_size()}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        shuffle=True,
        num_replicas=get_world_size(),
        rank=get_rank(),
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        shuffle=False,
        num_replicas=get_world_size(),
        rank=get_rank(),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
    )

    ## Setup mixup / cutmix 
    mixup_fn = None
    mixup_active = config.aug.mixup > 0 or config.aug.cutmix > 0. \
        or config.aug.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=config.aug.mixup, 
                         cutmix_alpha=config.aug.cutmix,
                         cutmix_minmax=config.aug.cutmix_minmax,
                         prob=config.aug.mixup_prob,
                         switch_prob=config.aug.mixup_switch_prob,
                         mode=config.aug.mixup_mode,
                         label_smoothing=config.aug.label_smoothing,
                         num_classes=config.num_classes)
    return train_dataset, val_dataset, train_loader, val_loader, mixup_fn


def create_xview_transforms():
    transforms = []
    return A.Compose(
        [],
        additional_targets={
            "conf_mask": "mask",
            "length_mask": "mask",
            "vessel_mask": "mask",
            "fishing_mask": "mask",
            "center_mask": "mask",
        },
    )

def create_imagenet_transforms(config: DataConfig, is_train: bool = True):
    resize_im = config.crop_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.crop_size,
            is_training=True,
            color_jitter=config.aug.color_jitter
            if config.aug.color_jitter > 0 else None,
            auto_augment=config.aug.auto_augment
            if not config.aug.auto_augment is None else None,
            re_prob=config.aug.reprob,
            re_mode=config.aug.remode,
            re_count=config.aug.recount,
            interpolation=config.interpolation,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                config.crop_size, padding=4)

        return transform

    t = []
    if resize_im:
        if config.test_crop:
            size = int(1.0 * config.crop_size)
            t.append(
                transforms.Resize(size,
                                  interpolation=_pil_interp(
                                      config.interpolation)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.crop_size))
        elif config.aug.random_resized_crop:
            t.append(
                transforms.RandomResizedCrop(
                    (config.crop_size, config.crop_size),
                    interpolation=_pil_interp(config.interpolation)))
        else:
            t.append(
                transforms.Resize(
                    (config.crop_size, config.crop_size),
                    interpolation=_pil_interp(config.interpolation)))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(config.aug.mean, config.aug.std))

    return transforms.Compose(t)


class TestDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.names = os.listdir(root_dir)

    def __getitem__(self, index):
        return dict(name=self.names[index])

    def __len__(self):
        return len(self.names)
