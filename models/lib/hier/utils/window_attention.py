import math
from itertools import product
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .patch_embed import compute_downsample_pad, crop_4d


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size[0],
        window_size[0],
        W // window_size[1],
        window_size[1],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_partition4d(x, window_size: Tuple[int, int]):
    """
    Args:
        x: ( N C T L H W)
        window_size (int): window size

    Returns:
        windows:
    """
    return rearrange(
        x,
        "N C (T WT) (L WL) (H WH) (W WW) -> (N T L H W )  (WT WL WH WW ) C",
    )


def window_reverse(
    windows, window_size: Tuple[int, int], img_size: Tuple[int, int]
):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(
        -1,
        H // window_size[0],
        W // window_size[1],
        window_size[0],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowAttention4D(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        input_resolution,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        roll=False,
        bias_mode="swin",
        mask=True,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.shift_size = np.array([x // 2 for x in self.window_size]).astype(
            int
        )
        self.roll = roll
        self.num_heads = num_heads
        self.original_input_resolution = input_resolution
        self.input_resolution = np.array(input_resolution)
        # compte padd

        self.padding = compute_downsample_pad(
            self.input_resolution, self.window_size, one_sided=True
        )
        self.actual_padding = (
            np.concatenate([[0, v] for v in reversed(self.padding)])
            .astype(int)
            .tolist()
        )
        # self.actual_padding = np.ceil(paddings / 2.0)
        self.do_padding = sum(self.padding) > 0
        self.input_resolution = self.input_resolution + self.padding
        self.input_resolution = self.input_resolution.astype(int)

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1)))
        )

        # mlp to generate continuous relative position bias

        # get relative_coords_table
        self.window_parms = self.get_window_parms()
        self.window_area = np.product(self.window_size)

        self.bias_mode = bias_mode
        if self.bias_mode == "swin":
            self.cpb_mlp = nn.Sequential(
                nn.Linear(len(self.window_size), 512, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_heads, bias=False),
            )
            relative_coords = [
                torch.arange(-(x - 1), x, dtype=torch.float32)
                for x in self.window_size
            ]
            # relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
            # relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
            relative_coords_table = (
                torch.stack(torch.meshgrid(relative_coords))
                .permute(*range(1, len(self.window_size) + 1), 0)
                .contiguous()
                .unsqueeze(0)
            )  # 1, 2*Wh-1, 2*Ww-1, 2
            for i, v in enumerate(self.window_size):
                relative_coords_table[..., i] /= v - 1 + 1e-9
            relative_coords_table *= 8  # normalize to -8, 8
            relative_coords_table = (
                torch.sign(relative_coords_table)
                * torch.log2(torch.abs(relative_coords_table) + 1.0)
                / math.log2(8)
            )

            self.register_buffer(
                "relative_coords_table", relative_coords_table, persistent=False
            )
            # get pair-wise relative position index for each token inside the window
            coords = [torch.arange(x) for x in self.window_size]
            # coords_h = torch.arange(self.window_size[0])
            # coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid(coords))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = (
                coords_flatten[:, :, None] - coords_flatten[:, None, :]
            )  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0
            ).contiguous()  # Wh*Ww, Wh*Ww, 2
            for idx, v in enumerate(self.window_size):
                relative_coords[:, :, idx] += v - 1  # shift to start from 0
            for i, v in enumerate(self.window_size):
                for j in range(0, i):
                    relative_coords[:, :, j] *= 2 * self.window_size[i] - 1
            # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            # relative_coords[:, :, 1] += self.window_size[1] - 1
            # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer(
                "relative_position_index",
                relative_position_index,
                persistent=False,
            )
        elif self.bias_mode in ["earth", "pangu"]:
            T, L, H, W = self.window_size
            NT, NL, NH, NW = self.input_resolution
            nL = np.product(self.window_size)
            # target should be NW * (TLHW) ( TLHW)
            abs_coords = [
                torch.arange(x, dtype=torch.float32)
                for x in self.input_resolution
            ]  # T L H W
            abs_coords = torch.stack(
                torch.meshgrid(abs_coords), dim=-1
            ).contiguous()  # T L H W 4
            for i, v in enumerate(self.input_resolution):
                abs_coords[..., i] /= v - 1 + 1e-9
            abs_coords = (abs_coords - 0.5) * 2
            # abs_coords *= 8  # normalize to -8, 8
            abs_coords = rearrange(
                abs_coords,
                "(NT WT) (NL WL) (NH WH) (NW WW) C -> (NT NL NH NW)   (WT WL WH WW ) C",
                **self.window_parms
            )
            abs_coords_x = abs_coords[:, :, None].repeat([1, 1, nL, 1])
            abs_coords_y = abs_coords[
                :,
                None,
            ].repeat([1, nL, 1, 1])
            if self.bias_mode == "earth":
                abs_table = torch.cat(
                    [abs_coords_x, abs_coords_y], dim=-1
                )  # (NT NL NH NW)   (WT WL WH WW )  (WT WL WH WW ) C
            elif self.bias_mode == "pangu":
                abs_table = torch.stack(
                    [
                        abs_coords_x[..., 0] - abs_coords_y[..., 0],
                        abs_coords_x[..., 1],
                        abs_coords_y[..., 1],
                        abs_coords_x[..., 2],
                        abs_coords_y[..., 2],
                        abs_coords_x[..., 3] - abs_coords_y[..., 3],
                    ],
                    dim=-1,
                )  # (NT NL NH NW)   (WT WL WH WW )  (WT WL WH WW ) C
            else:
                raise NotImplementedError
            self.cpb_mlp = nn.Sequential(
                nn.Linear(abs_table.shape[-1], 512, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_heads, bias=False),
            )
            self.register_buffer("abs_table", abs_table, persistent=False)
        else:
            raise NotImplementedError
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.register_buffer("k_bias", torch.zeros(dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.mask = mask
        if self.mask:
            attn_mask = self.build_atten_mask()
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    def get_window_parms(self):
        WT, WL, WH, WW = self.window_size
        return dict(WT=WT, WL=WL, WH=WH, WW=WW)

    def build_atten_mask(self):
        if self.roll:
            # calculate attention mask for SW-MSA
            # H, W = self.input_resolution
            img_mask = torch.zeros((1, *self.input_resolution, 1))  # 1 H W 1
            cnt = 0
            shifts_iters = [
                (
                    slice(0, -self.window_size[i]),
                    slice(-self.window_size[i], -self.shift_size[i]),
                    slice(-self.shift_size[i], None),
                )
                for i in range(len(self.window_size))
            ]
            for i, j, k, r in product(*shifts_iters):
                img_mask[:, i, j, k, r, :] = cnt
                cnt += 1
            # for h in (
            #         slice(0, -self.window_size[0]),
            #         slice(-self.window_size[0], -self.shift_size[0]),
            #         slice(-self.shift_size[0], None)):
            #     for w in (
            #             slice(0, -self.window_size[1]),
            #             slice(-self.window_size[1], -self.shift_size[1]),
            #             slice(-self.shift_size[1], None)):
            #         img_mask[:, h, w, :] = cnt
            #         cnt += 1
            # mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = rearrange(
                img_mask,
                "N  (NT WT) (NL WL) (NH WH) (NW WW) C -> (N NT NL NH NW )  (WT WL WH WW ) C",
                **self.window_parms
            )
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def transform(self, x):
        if self.do_padding:
            x = F.pad(x, self.actual_padding)
        if self.roll:
            x = torch.roll(x, shifts=tuple(-self.shift_size), dims=(2, 3, 4, 5))
        N_0, C, T, L, H, W = x.shape
        x = rearrange(
            x,
            "N C (NT WT) (NL WL) (NH WH) (NW WW) -> (N NT NL NH NW )  (WT WL WH WW ) C",
            **self.window_parms
        )  # B_
        WT, WL, WH, WW = self.window_size
        NT = T // WT
        NL = L // WL
        NH = H // WH
        NW = W // WW
        return x, dict(NT=NT, NL=NL, NH=NH, NW=NW)

    def inverse_transform(self, x, transform_info):
        x = rearrange(
            x,
            "(N NT NL NH NW )  (WT WL WH WW ) C -> N C (NT WT) (NL WL) (NH WH) (NW WW)",
            **self.window_parms,
            **transform_info
        )  # B_
        if self.roll:
            x = torch.roll(x, shifts=tuple(self.shift_size), dims=(2, 3, 4, 5))
        if self.do_padding:
            x = crop_4d(x, self.original_input_resolution)
        return x

    def forward(self, x):
        x0_shape = x.shape
        x, transform_info = self.transform(x)
        x = self._atten(x, mask=self.attn_mask)
        x = self.inverse_transform(x, transform_info)
        assert x0_shape == x.shape
        return x

    def _atten(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(
            self.logit_scale, max=math.log(1.0 / 0.01)
        ).exp()
        attn = attn * logit_scale
        if self.bias_mode == "swin":
            relative_position_bias_table = self.cpb_mlp(
                self.relative_coords_table
            ).view(-1, self.num_heads)
            relative_position_bias = relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                np.product(self.window_size), np.product(self.window_size), -1
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
            attn = attn + relative_position_bias.unsqueeze(0)
        elif self.bias_mode in ["earth", "pangu"]:
            nL = np.product(self.window_size)
            nH = self.num_heads
            abs_position_bias = self.cpb_mlp(self.abs_table).view(
                -1, nL, nL, nH
            )  # nW * Wh*Ww, Wh*Ww * nH
            abs_position_bias = abs_position_bias.permute(
                0, 3, 1, 2
            )  # nW nH, Wh*Ww, Wh*Ww
            abs_position_bias = 16 * torch.sigmoid(abs_position_bias)
            nW = abs_position_bias.shape[0]
            n_batch = B_ // abs_position_bias.shape[0]
            attn = attn.view(
                n_batch, nW, nH, nL, nL
            ) + abs_position_bias.unsqueeze(0)
            attn = attn.view((n_batch * nW), nH, nL, nL)
        else:
            raise NotImplemented

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(
                -1, num_win, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = self.inverse_transform(x)
        return x
