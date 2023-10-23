import math
import os
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset


def slide_name_to_patient_barcode(slide_name):
    """
    Take a slide filepath and return the corresponding patient barcode
    """
    return "-".join(slide_name.stem.split("-")[:3])


class TCGADataset(Dataset):
    def __init__(
        self,
        mode: str = "train",
        dataset_dir: str = "/shared/ritwik/data/tcga/",
        annotation_csv: str = "mmc1.xlsx",
        folds_csv: str = "/shared/ritwik/data/tcga/splits/splits_0.csv",
        crop_size: int = 1024,
        transforms: A.Compose = None,
    ):
        """
        Args:
            mode: Can be either "train" or "val"
        """
        self.dataset_dir = dataset_dir
        self.folds = pd.read_csv(folds_csv)
        self.labels = pd.read_csv(annotation_csv)
        self.files = pd.Series(Path(self.dataset_dir).glob("tcga_*/WSIs/*.svs"))

        self.mode = mode
        if self.mode not in ["train", "val"]:
            raise NotImplementedError("Please provide a valid mode ('train', 'val').")
        self.labels = self._process_labels(self.folds, self.labels)

        self.transforms = transforms
        self.crop_size = crop_size
        if self.mode == "train":
            # Shuffle the rows of the dataframe
            self.files = self.files.sample(frac=1).reset_index(drop=True)


    def _process_labels(self, folds, labels):
        """
        Only keep the files specified by the folds and make them into a lookup table
        """
        # Filter on splits
        # Drop the first index column of both folds and labels
        labels = pd.merge(folds.iloc[:, 1:], labels.iloc[:, 1:], how="inner", left_on=self.mode, right_on="bcr_patient_barcode")

        if self.mode == "train":
            labels = labels.drop("val", axis=1)
        elif self.mode == "val":
            labels = labels.drop("train", axis=1)
        
        # This column is redundant
        labels = labels.drop("bcr_patient_barcode", axis=1)

        # Convert into a lookup table
        labels = labels.rename(columns={self.mode: "barcode"})
        labels = labels.set_index("barcode").to_dict("index")
        return labels


    def __getitem__(self, i):
        pass