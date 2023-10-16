from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TileSlice:
    row: int
    column: int
    y: int
    x: int


class Tiler:
    def __init__(
        self,
        height: int,
        width: int,
        tile_size: int = 1536,
        overlap: int = 512,
        pad_value=0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.tile_size = tile_size
        self.overlap = overlap
        self.pad_value = pad_value

    def generate_slices(self) -> List[TileSlice]:
        stride = self.tile_size - self.overlap
        rows = self.height // stride + 1
        cols = self.width // stride + 1
        slices = []
        for row in range(rows):
            for column in range(cols):
                start_y = row * stride
                start_x = column * stride
                if start_x > self.width or start_y > self.height:
                    continue
                slices.append(TileSlice(row, column, start_y, start_x))
        return slices

    def generate_crops(self, img):
        slices = self.generate_slices()
        crops_and_slices = []
        for slice in slices:
            crop = img[
                :,
                slice.y : slice.y + self.tile_size,
                slice.x : slice.x + self.tile_size,
            ].astype(np.float16)
            c, c_h, c_w = crop.shape
            if c_h < self.tile_size or c_w < self.tile_size:
                tmp = np.zeros((c, self.tile_size, self.tile_size), dtype=np.float16)
                tmp[:, :, :] = self.pad_value
                tmp[:, :c_h, :c_w] = crop
                crop = tmp
            crops_and_slices.append((crop, slice))
        return crops_and_slices

    def combine_crops(self, preds_slices, dtype=np.float16):
        out = np.zeros((self.height, self.width), dtype)
        for pred, slice in preds_slices:
            out_crop = out[
                slice.y : slice.y + self.tile_size, slice.x : slice.x + self.tile_size
            ]
            left_pad = self.overlap // 2 if slice.column > 0 else 0
            top_pad = self.overlap // 2 if slice.row > 0 else 0
            out_crop = out_crop[top_pad:, left_pad:]
            if out_crop.shape[0] > 0 and out_crop.shape[1] > 0:
                out_crop[:, :] = pred[top_pad:, left_pad:][
                    : out_crop.shape[0], : out_crop.shape[1]
                ]
        return out

    def update_crop(self, out, pred, slice):
        out_crop = out[
            slice.y : slice.y + self.tile_size, slice.x : slice.x + self.tile_size
        ]
        left_pad = self.overlap // 2 if slice.column > 0 else 0
        top_pad = self.overlap // 2 if slice.row > 0 else 0
        out_crop = out_crop[top_pad:, left_pad:]
        if out_crop.shape[0] > 0 and out_crop.shape[1] > 0:
            out_crop[:, :] = pred[top_pad:, left_pad:][
                : out_crop.shape[0], : out_crop.shape[1]
            ]
        return out

    def get_crop(self, img, slice):
        crop = img[
            slice.y : slice.y + self.tile_size, slice.x : slice.x + self.tile_size
        ].astype(np.float16)
        c_h, c_w = crop.shape
        if c_h < self.tile_size or c_w < self.tile_size:
            tmp = np.zeros((self.tile_size, self.tile_size), dtype=np.float16)
            tmp[:, :] = self.pad_value
            tmp[:c_h, :c_w] = crop
            crop = tmp
        return crop
