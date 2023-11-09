import json
from pathlib import Path
from typing import Dict, List, Set, Union

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
        channels_first: bool = False,
        categories_json: str = "categories.json",
        supercategories: List[str] = None,
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
        self.categories = json.load(open(self.dataset_dir / categories_json))
        self.supercategories = set(supercategories)
        self.labels, self.category_label_map = self._process_labels(self.labels, self.supercategories)
        self.label_category_map = {v: k for k, v in self.category_label_map.items()}
        self.channels_first = channels_first

        print(self.label_category_map)

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
        img = np.asarray(Image.open(img_path).convert('RGB'))
        if self.transforms:
            img = self.transforms(image=img)["image"]
        if self.channels_first:
            img = img.transpose((2, 0, 1))

        return {"image": img, **label}

    def _process_labels(self, labels: COCO, supercategories: Set[str]):
        """
        Load keys, images, and annotations
        """
        ids = sorted(list(labels.anns.keys()))
        valid_category_ids = set([x["id"] for x in self.categories if x["supercategory"] in supercategories])
        category_label_map = {}

        ret_labels = []
        counter = 0
        for id in ids:
            label = labels.anns[id]["category_id"]
            if len(supercategories) > 0 and label not in valid_category_ids:
                continue

            if label not in category_label_map:
                category_label_map[label] = counter
                counter += 1

            ret_labels.append({
                "id": id,
                "file_name": labels.imgs[id]["file_name"],
                "label": category_label_map[label],
                "id": labels.imgs[id]["id"],
            })

        return ret_labels, category_label_map
    
    def output_to_category(self, model_output):
        return self.label_category_map[model_output]