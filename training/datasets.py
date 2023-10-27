import math
import os
import random
import time
from dataclasses import dataclass

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import rasterio
import tifffile
import torch

# from hydra.core.config_store import ConfigStore
from rasterio.windows import Window
from torch.utils.data import Dataset

from .utils import is_main_process

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


# VV mean: -15.830463789539426
# VV std:  6.510123043441801

# VH mean: -24.66130160959856
# VH std:  6.684547156770566

train_transforms = A.Compose(
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


# cs = ConfigStore.instance()
# cs.store(name="config", group="data", node=DataConfig)


def normalize_band(band, ignored_mask=0):
    band[band < -32760] = -100
    ignored_idx = band == -100
    if np.count_nonzero(band != -100) == 0:
        band[:, :] = ignored_mask
    else:
        band = (band + 40) / 15
        band[ignored_idx] = ignored_mask
    return band


def create_data_datasets(config: DataConfig):
    if is_main_process():
        print("dataset config crop size", config.crop_size)
        if config.shoreline_dir:
            print("Legacy Warning:shoreline_dir is no longer used")
    if config.test:
        train_annotations = os.path.join(config.dir, "labels/public.csv")
        train_dataset = XviewValDataset(
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
        train_annotations = os.path.join(config.dir, "labels/validation.csv")
        train_dataset = XviewValDataset(
            mode="train",
            dataset_dir=config.dir,
            fold=config.fold,
            folds_csv=config.folds_csv,
            annotation_csv=train_annotations,
            crop_size=config.crop_size,
            multiplier=config.multiplier,
            positive_ratio=config.positive_ratio,
        )
        val_dataset = XviewValDataset(
            mode="val",
            dataset_dir=config.dir,
            fold=config.fold,
            folds_csv=config.folds_csv,
            annotation_csv=train_annotations,
            crop_size=config.val_crop_size,
        )
    return train_dataset, val_dataset


class TestDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.names = os.listdir(root_dir)

    def __getitem__(self, index):
        return dict(name=self.names[index])

    def __len__(self):
        return len(self.names)


class XviewValDataset(Dataset):
    def __init__(
        self,
        mode: str,
        dataset_dir: str,
        annotation_csv: str,
        folds_csv: str,
        multiplier: int = 1,
        fold: int = 0,
        crop_size: int = 1024,
        sigma: int = 2,
        radius: int = 4,
        transforms: A.Compose = train_transforms,
        positive_ratio=0.5,
    ):
        df = pd.read_csv(folds_csv)
        self.radius = radius

        if mode == "train":
            self.names = df[df.fold != fold].scene_id.tolist()
        else:
            self.names = df[df.fold == fold].scene_id.tolist()
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.df = pd.read_csv(annotation_csv)
        self.crop_size = crop_size
        self.sigma = sigma
        self.names = multiplier * self.names
        self.positive_ratio = positive_ratio
        if self.mode == "train":
            random.shuffle(self.names)

    def __getitem__(self, i):
        if self.mode == "val":
            return {
                "name": self.names[i],
            }
        rm = random.Random()
        rm.seed(time.time_ns())
        name = self.names[i]
        crop_size = self.crop_size

        vv_full = rasterio.open(
            os.path.join(
                self.dataset_dir, "images/validation", name, "VV_dB.tif"
            )
        )
        vh_full = rasterio.open(
            os.path.join(
                self.dataset_dir, "images/validation", name, "VH_dB.tif"
            )
        )
        h, w = vv_full.shape

        df = self.df
        df = df[df.scene_id == name]
        points = [row for _, row in df.iterrows()]
        if len(points) > 1 and random.random() > (1.0 - self.positive_ratio):
            point_idx = rm.randint(0, len(points) - 1)
            point = points[point_idx]
            y, x = point.detect_scene_row, point.detect_scene_column
            max_shift_pad = 32
            min_x_start = min(
                max(x - crop_size + max_shift_pad, 0), w - crop_size - 32
            )
            min_y_start = min(
                max(y - crop_size + max_shift_pad, 0), h - crop_size - 32
            )
            max_x_start = max(min(x - max_shift_pad, w - crop_size - 1), 0)
            max_y_start = max(min(y - max_shift_pad, h - crop_size - 1), 0)

            if max_x_start < min_x_start:
                min_x_start, max_x_start = max_x_start, min_x_start
            if max_y_start < min_y_start:
                min_y_start, max_y_start = max_y_start, min_y_start
            h_start = rm.randint(int(min_y_start), int(max_y_start))
            w_start = rm.randint(int(min_x_start), int(max_x_start))
            h_end = h_start + crop_size
            w_end = w_start + crop_size
            # vh = vh_full[h_start: h_end, w_start: w_end].astype(np.float32)
            # vv = vv_full[h_start: h_end, w_start: w_end].astype(np.float32)
            vh = vh_full.read(
                1,
                window=Window(
                    w_start, h_start, w_end - w_start, h_end - h_start
                ),
            )
            vv = vv_full.read(
                1,
                window=Window(
                    w_start, h_start, w_end - w_start, h_end - h_start
                ),
            )
        else:
            for i in range(5):
                h_start = rm.randint(0, h - crop_size - 1)
                w_start = rm.randint(0, w - crop_size - 1)

                h_end = h_start + crop_size
                w_end = w_start + crop_size
                # vh = vh_full[h_start: h_end, w_start: w_end].astype(np.float32)
                vh = vh_full.read(
                    1,
                    window=Window(
                        w_start, h_start, w_end - w_start, h_end - h_start
                    ),
                )
                vv = vv_full.read(
                    1,
                    window=Window(
                        w_start, h_start, w_end - w_start, h_end - h_start
                    ),
                )
                known_pixels = np.count_nonzero(vh > -1000)
                # vv = vv_full[h_start: h_end, w_start: w_end].astype(np.float32)
                if known_pixels / (crop_size * crop_size) > 0.05:
                    break
        vh_full.close()
        vv_full.close()
        object_mask = np.zeros_like(vv, dtype=np.float32)
        vessel_mask = np.zeros_like(vv, dtype=np.float32)
        fishing_mask = np.zeros_like(vv, dtype=np.float32)
        conf_mask = np.zeros_like(vv, dtype=np.float32)

        length_mask = np.zeros_like(vv)
        length_mask[:, :] = -1
        center_mask = np.zeros_like(vv)
        size = 6 * self.sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))
        crop_coords = np.zeros((1024, 4))
        crop_coords_idx = 0
        for _, row in df.iterrows():
            if (
                h_start < row.detect_scene_row < h_end
                and w_start < row.detect_scene_column < w_end
            ):
                x = row.detect_scene_column - w_start
                y = row.detect_scene_row - h_start

                # CENTER MASK
                # upper left
                ul = int(np.round(x - 3 * self.sigma - 1)), int(
                    np.round(y - 3 * self.sigma - 1)
                )
                # bottom right
                br = int(np.round(x + 3 * self.sigma + 2)), int(
                    np.round(y + 3 * self.sigma + 2)
                )

                c, d = max(0, -ul[0]), min(br[0], self.crop_size) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.crop_size) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.crop_size)
                aa, bb = max(0, ul[1]), min(br[1], self.crop_size)
                center_mask[aa:bb, cc:dd] = np.maximum(
                    center_mask[aa:bb, cc:dd], g[a:b, c:d]
                )
                # DEFINE VESSELS
                # man-made maritime object
                object_cls = 1
                vessel_cls = 0
                fishing_cls = 0
                if math.isnan(row.is_vessel):
                    vessel_cls = 255
                elif row.is_vessel:
                    vessel_cls = 1

                if vessel_cls == 0:
                    fishing_cls = 0
                elif math.isnan(row.is_fishing):
                    fishing_cls = 255
                elif row.is_fishing:
                    fishing_cls = 1
                confs = ["none", "LOW", "MEDIUM", "HIGH"]
                conf_idx = confs.index(row.confidence)
                if conf_idx > 1:
                    conf_idx = 2
                cv2.circle(
                    conf_mask,
                    center=(x, y),
                    radius=self.radius,
                    color=conf_idx,
                    thickness=-1,
                )
                cv2.circle(
                    object_mask,
                    center=(x, y),
                    radius=self.radius if object_cls < 200 else 7,
                    color=object_cls,
                    thickness=-1,
                )
                cv2.circle(
                    vessel_mask,
                    center=(x, y),
                    radius=self.radius if vessel_cls < 200 else 7,
                    color=vessel_cls,
                    thickness=-1,
                )
                cv2.circle(
                    fishing_mask,
                    center=(x, y),
                    radius=self.radius if fishing_cls < 200 else 7,
                    color=fishing_cls,
                    thickness=-1,
                )
                # length MASK
                vessel_length = -1
                if not math.isnan(row.vessel_length_m):
                    vessel_length = row.vessel_length_m
                cv2.circle(
                    length_mask,
                    center=(x, y),
                    radius=self.radius if vessel_length > 0 else 7,
                    color=vessel_length,
                    thickness=-1,
                )
                if conf_idx > 1:
                    pad = 9
                    y1, y2 = y - pad, y + pad
                    x1, x2 = x - pad, x + pad
                    if (
                        x1 > 32
                        and x2 < self.crop_size - 32
                        and y1 > 32
                        and y2 < self.crop_size - 32
                    ):
                        crop_coords[crop_coords_idx] = np.array(
                            [x1, y1, x2, y2]
                        )
                        crop_coords_idx += 1

        vv = normalize_band(band=vv, ignored_mask=0)
        vh = normalize_band(band=vh, ignored_mask=0)
        image = np.stack([vv, vh], axis=-1).astype(np.float32)
        sample = self.transforms(
            image=image,
            mask=object_mask,
            center_mask=center_mask,
            length_mask=length_mask,
            conf_mask=conf_mask,
            fishing_mask=fishing_mask,
            vessel_mask=vessel_mask,
        )
        image = sample["image"]
        object_mask = sample["mask"]
        center_mask = sample["center_mask"]
        length_mask = sample["length_mask"]
        vessel_mask = sample["vessel_mask"]
        fishing_mask = sample["fishing_mask"]
        conf_mask = sample["conf_mask"]
        image = torch.from_numpy(image).float().moveaxis(-1, 0)
        center_mask = torch.from_numpy(center_mask).float().unsqueeze(0) * 255
        length_mask = torch.from_numpy(length_mask).float().unsqueeze(0)
        conf_mask = torch.from_numpy(conf_mask).long()
        object_mask = torch.from_numpy(object_mask).float().unsqueeze(0)
        vessel_mask = torch.from_numpy(vessel_mask).float().unsqueeze(0)
        fishing_mask = torch.from_numpy(fishing_mask).float().unsqueeze(0)

        if random.random() < 0.5:
            # 180 rotate to handle different sar orientation
            image = torch.rot90(image, 2, dims=(1, 2))
            center_mask = torch.rot90(center_mask, 2, dims=(1, 2))
            length_mask = torch.rot90(length_mask, 2, dims=(1, 2))
            conf_mask = torch.rot90(conf_mask, 2, dims=(0, 1))
            object_mask = torch.rot90(object_mask, 2, dims=(1, 2))
            vessel_mask = torch.rot90(vessel_mask, 2, dims=(1, 2))
            fishing_mask = torch.rot90(fishing_mask, 2, dims=(1, 2))
            ori_crops = crop_coords.copy()
            crop_coords = self.crop_size - crop_coords
            crop_coords[ori_crops == 0] = 0
            crop_coords = crop_coords[:, [2, 3, 0, 1]]
        crop_coords = torch.from_numpy(crop_coords).long()

        return {
            "image": image,
            "object_mask": object_mask,
            "crop_coords": crop_coords,
            "conf_mask": conf_mask,
            "vessel_mask": vessel_mask,
            "fishing_mask": fishing_mask,
            "center_mask": center_mask,
            "length_mask": length_mask,
            "name": name,
        }

    def __len__(self):
        return len(self.names)
