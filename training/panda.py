import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from openslide import OpenSlide
from PIL import Image
from torch.utils.data import Dataset


class Panda(Dataset):
    """PANDA Prostate dataset."""

    radboud_to_karolinska = {
        0: 0,  # background -> background
        1: 1,  # stroma -> benign
        2: 1,  # benign epithelium -> benign
        3: 2,  # Gleason 3 -> cancerous
        4: 2,  # Gleason 4 -> cancerous
        5: 2,  # Gleason 5 -> cancerous
    }

    gleason_to_isup = {
        # majority + minority -> ISUP
        "0+0": 0,
        "3+3": 1,
        "3+4": 2,
        "4+3": 3,
        "4+4": 4,
        "3+5": 4,
        "5+3": 4,
        "4+5": 5,
        "5+4": 5,
        "5+5": 5,
    }

    inverse_idx_gleason = {
        k: idx for idx, k in enumerate(gleason_to_isup.keys())
    }
    idx_gleason = {idx: k for idx, k in enumerate(gleason_to_isup.keys())}

    def __init__(
        self,
        root_dir="/shared/ritwik/data/panda-prostate/",
        split="train",
        split_file="/shared/ritwik/data/panda-prostate/train_mask_intersect.csv",
        transform=None,
        load_mask=True,
        crop_size=1024,
        stride=None,
        mode="random",
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            split_file (string): Data file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(root_dir) == Path:
            root_dir = str(root_dir)

        if stride is None:
            stride = crop_size

        self.root_dir = root_dir
        self.split = split
        self.files = self._load_file_info(split_file)
        self.length = self.files.count()

        self.transform = transform
        self.load_mask = load_mask
        self.crop_size = crop_size
        self.stride = stride
        self.mode = mode

        self.chips = None
        if self.mode == "slice":
            chips = []
            print(f"Chipping {len(self.files)} files...")
            for idx, (i, file) in enumerate(self.files.iterrows()):
                filepath = f'{self.root_dir}/{self.split}_images/{file["image_id"]}.tiff'
                img = OpenSlide(filepath)
                h, w = img.dimensions
                img.close()
                i = 0
                j = 0
                while i < h:
                    while j < w:
                        chips.append()
                        j += self.stride
                        payload = (idx, i, j, self.crop_size, self.crop_size)
                        chips.append(payload)
                    i += self.stride
            self.chips = chips
            print("DONE")
        elif self.mode == "random":
            pass
        elif self.mode == "full":
            chips = []
            print(f"Chipping {len(self.files)} files...")
            for idx, (i, file) in enumerate(self.files.iterrows()):
                filepath = f'{self.root_dir}/{self.split}_images/{file["image_id"]}.tiff'
                img = OpenSlide(filepath)
                h, w = img.dimensions
                payload = (idx, 0, 0, h, w)
                chips.append(payload)
                img.close()
            print("DONE")
            self.chips = chips
        else:
            raise NotImplemented

    def _load_file_info(self, split_file):
        df = pd.read_csv(split_file)
        # Only load Radboud data files as they have fully labeled masks
        df = df[df["data_provider"] == "radboud"]

        return df

    def __len__(self):
        if self.chips is not None:
            return len(self.chips)
        return len(self.files)

    def __getitem__(self, idx):
        assert type(idx) == int, f"Invalid Index {idx,type(idx)}"

        if self.chips is not None:
            idx, i, j, h, w = self.chips[idx]
        elif self.mode == "random":
            pass
        else:
            raise NotImplemented
        file = self.files.iloc[idx]
        filepath = (
            f'{self.root_dir}/{self.split}_images/{file["image_id"]}.tiff'
        )

        img = OpenSlide(filepath)  # img size is obtained via img.dimensions)
        if self.mode == "random":
            h_max, w_max = img.dimensions
            i = np.random.rand() * (h_max - self.crop_size)
            j = np.random.rand() * (w_max - self.crop_size)
            i = int(i)
            j = int(j)
            h = w = self.crop_size

        pixel_spacing = 1 / (
            float(img.properties["tiff.XResolution"]) / 10000
        )  # microns
        # print(h,w)
        img_handler = img
        img = img.read_region(
            location=(i, j),  # The coordinate of the top left for the read
            level=0,  # 0 is the highest resolution,
            size=(
                h,
                w,
            ),  # The amount of data to read TODO replace this with the chip size
        ).convert(
            "RGB"
        )  # By default it is RGBA
        img = np.array(img).transpose(2, 0, 1)  # C, H, W
        img_handler.close()
        if self.load_mask and file["has_mask"]:
            mask = self._load_mask(file, i, j, h, w)
        else:
            mask = None

        isup_grade = file["isup_grade"]
        gleason_score = file["gleason_score"]
        if (
            gleason_score == "negative"
        ):  # 0+0 Gleason score is equal to negative
            gleason_score = "0+0"

        if self.transform:
            # img = self.transform(img)
            # print(img.shape,mask.shape,img.dtype,mask.dtype)
            img = torch.tensor(img).float().unsqueeze(0)

            if mask is not None:
                mask = torch.tensor(mask).long()
                mask = (
                    torch.nn.functional.one_hot(mask)
                    .permute(2, 0, 1)
                    .float()
                    .unsqueeze(0)
                )
                img, mask = self.transform(img, mask)
                img = img.squeeze(0)
                mask = mask.squeeze(0)
                mask = mask.argmax(0)
            else:
                img = self.transform(img)
            # TODO you must crop the mask too!
            # i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(128, 128))
            # img = F.crop(img, i, j, h, w)
            # mask = F.crop(mask, i, j, h, w)

            # print(img.shape,mask.shape,img.dtype,mask.dtype)
        local_score = self._assign_patch_gleason(mask)
        local_score = self.inverse_idx_gleason[local_score]
        return {
            "image": img,
            "mask": mask,
            "isup_grade": isup_grade,
            "gleason_score": gleason_score,
            "local_score": local_score,
            "pixel_spacing": pixel_spacing,
        }

    def _load_mask(self, file, i, j, h, w):
        mask_path = f'{self.root_dir}/{self.split}_label_masks/{file["image_id"]}_mask.tiff'
        mask_handler = OpenSlide(mask_path)
        mask = mask_handler.read_region(
            location=(i, j), level=0, size=(h, w)
        )  # TODO replace with chip read
        mask_handler.close()
        mask = mask.split()[0]  # The mask data is in the 'R' channel
        mask = np.array(mask)  # H X W

        # if file['data_provider'] == 'radboud':
        #    mask = np.vectorize(self.radboud_to_karolinska.get)(mask)

        return mask

    @classmethod
    def isup_grade_from_mask(cls, patch):
        return cls.gleason_to_isup[cls._assign_patch_gleason(patch)]

    @classmethod
    def isup_grade_from_local_score(cls, local_scores):
        """
        local_scores: List[int], List of int label of local_score, which can be mapped to string using cls.idx_gleason
        """
        return 1

    @staticmethod
    def _assign_patch_gleason(patch):
        """Patch must be provided in Radboud mask format"""
        """
        path: Tensor[H X W]
        """
        if isinstance(patch, torch.Tensor):
            patch = patch.numpy()
        uniques, counts = np.unique(patch, return_counts=True)
        h, w = patch.shape
        total_pixels = h * w
        uniq_counts = list(zip(uniques.tolist(), counts.tolist()))
        # Remove background from counts
        uniq_counts = [x for x in uniq_counts if x[0] != 0]
        uniq_percents = {x[0]: (x[1] / total_pixels) for x in uniq_counts}

        if len(uniq_percents) == 0:  # Only background in this patch
            majority = minority = 0
        elif len(uniq_percents) == 2 and set(uniq_percents.keys()) == {0, 1, 2}:
            # If there are only two types of tissue present, and that tissue is benign
            # then return negative
            majority = minority = 0
        elif len(uniq_percents) == 1:  # There's only one class in this patch
            majority = minority = int(next(iter(uniq_percents)))
        else:
            # There is cancer in this patch
            cancer_grades = sorted(
                [(x, uniq_percents.get(x, 0)) for x in [3, 4, 5]],
                key=lambda x: -x[1],
            )
            # The majority is the highest occurence cancer grade
            majority = cancer_grades[0][0]

            # The minority is the next highest occurence of grade IF it is >= 5% and 5 is not a valid grade
            if 5 in uniq_percents.keys():
                minority = 5
            elif cancer_grades[1][1] >= 0.05:
                minority = cancer_grades[1][0]
            else:
                minority = majority

        if majority in [1, 2]:
            majority = 0
        if minority in [1, 2]:
            minority = 0

        return f"{majority}+{minority}"
