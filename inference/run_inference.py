import gc
import os
from typing import List

import numpy as np
import tifffile
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from inference.tiling import Tiler, TileSlice
from training.val_dataset import normalize_band


class SliceDataset(Dataset):

    def __init__(self, scene_dir: str, tiler: Tiler) -> None:
        super().__init__()
        self.scene_dir = scene_dir
        self.tiler = tiler
        self.slices = tiler.generate_slices()
        self.vv_full = tifffile.imread(os.path.join(self.scene_dir, "VV_dB.tif"))
        self.vh_full = tifffile.imread(os.path.join(self.scene_dir, "VH_dB.tif"))

    def __getitem__(self, index):
        slice = self.slices[index]
        vv = self.tiler.get_crop(self.vv_full, slice)
        vh = self.tiler.get_crop(self.vh_full, slice)
        vv = normalize_band(vv)
        vh = normalize_band(vh)
        img = torch.from_numpy(np.stack([vv, vh], axis=0)).float()
        return img, np.array([slice.row, slice.column, slice.y, slice.x])

    def __len__(self):
        return len(self.slices)


def predict_scene_and_return_mm(models: List[nn.Module], dataset_dir, scene_id: str, use_fp16: bool = False,
                                rotate=False, output_dir=None):
    vh_full = tifffile.memmap(os.path.join(dataset_dir, scene_id, "VH_dB.tif"), mode="r")

    height, width = vh_full.shape

    tiler = Tiler(height, width, 3584, overlap=704)
    vessel_preds = np.zeros_like(vh_full, dtype=np.uint8)
    fishing_preds = np.zeros_like(vh_full, dtype=np.uint8)
    length_preds = np.zeros_like(vh_full, dtype=np.float16)
    center_preds = np.zeros_like(vh_full, dtype=np.uint8)
    slice_dataset = SliceDataset(os.path.join(dataset_dir, scene_id), tiler)
    slice_loader = DataLoader(
        slice_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False
    )
    for batch, slice_vals in tqdm(slice_loader):
        slice = TileSlice(*slice_vals[0])
        with torch.no_grad():
            batch = batch.cuda()
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = []
                for model in models:
                    output = model(batch)
                    sigmoid_keys = ["fishing_mask", "vessel_mask"]
                    for k in sigmoid_keys:
                        output[k] = torch.sigmoid(output[k])
                    if rotate:
                        out180 = model(torch.rot90(batch, 2, dims=(2, 3)))
                        for key in list(output.keys()):
                            val = torch.rot90(out180[key], 2, dims=(2, 3))
                            if key in sigmoid_keys:
                                val = torch.sigmoid(val)
                            output[key] += val
                            output[key] *= 0.5
                    outputs.append(output)
            output = {}
            for k in outputs[0].keys():
                vs = [o[k][:, :3] for o in outputs]
                output[k] = sum(vs) / len(models)
            vessel_mask = (output["vessel_mask"][0][0] * 255).cpu().numpy().astype(np.uint8)
            fishing_mask = (output["fishing_mask"][0][0] * 255).cpu().numpy().astype(np.uint8)
            center_mask = torch.clamp(output["center_mask"][0][0], 0, 255).cpu().numpy().astype(np.uint8)
            length_mask = output["length_mask"][0][0].cpu().numpy().astype(np.float16)
        tiler.update_crop(vessel_preds, vessel_mask, slice)
        tiler.update_crop(fishing_preds, fishing_mask, slice)
        tiler.update_crop(center_preds, center_mask, slice)
        tiler.update_crop(length_preds, length_mask, slice)
        # tiler.update_crop(conf_preds, conf_mask, slice)
    if output_dir:
        os.makedirs(os.path.join(output_dir, scene_id), exist_ok=True)
        np.save(os.path.join(output_dir, scene_id, "center_preds"), center_preds)
        np.save(os.path.join(output_dir, scene_id, "vessel_preds"), vessel_preds)
        np.save(os.path.join(output_dir, scene_id, "fishing_preds"), fishing_preds)
        np.save(os.path.join(output_dir, scene_id, "length_preds"), length_preds)
    gc.collect()
    return {
        "center_mask": center_preds,
        "vessel_mask": vessel_preds,
        "fishing_mask": fishing_preds,

        "length_mask": length_preds,
    }
