from pathlib import Path
from typing import Union

import albumentations as A
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class INatDataset(Dataset):
    def __init__(
        self,
        mode: str = "train",
        dataset_dir="/shared/ritwik/data/inaturalist2018/",
        annotation_json: str = "train2018.json",
        channels_first: bool = True,
        transforms: A.Compose = None,
    ):
        """
        Args:
            mode: Can be either "train" or "val"
        """
        if type(dataset_dir) is str:
            dataset_dir = Path(dataset_dir)
        self.dataset_dir = dataset_dir
        self.labels = COCO(annotation_file=str(dataset_dir / annotation_json))
        self.labels = self._process_labels(self.labels)
        self.channels_first = channels_first

        self.mode = mode
        if self.mode not in ["train", "val"]:
            raise NotImplementedError(
                "Please provide a valid mode ('train', 'val')."
            )

        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.dataset_dir / label["file_name"]
        img = np.asarray(Image.open(img_path))
        img = self.transforms(image=img)["image"]
        if self.channels_first:
            img = img.transpose((2, 0, 1))

        return {"image": img, **label}

    def _process_labels(self, labels: COCO):
        """
        Load keys, images, and annotations
        """
        ids = sorted(list(labels.anns.keys()))
        labels = [
            {
                "id": id,
                "file_name": labels.imgs[id]["file_name"],
                "label": labels.anns[id]["category_id"],
                "id": labels.imgs[id]["id"],
            }
            for id in ids
        ]

        return labels
