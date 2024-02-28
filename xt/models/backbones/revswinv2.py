""" Swin Transformer V2
A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
    - https://arxiv.org/abs/2111.09883

Code/weights from https://github.com/microsoft/Swin-Transformer, original copyright/license info below

Modifications and additions for timm hacked together by / Copyright 2022, Ross Wightman
"""
# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import math
import sys
from typing import Callable, Optional, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import (
    DropPath,  # manually add patchembed
    Format,
    Mlp,
    nchw_to,
    to_2tuple,
    trunc_normal_,
)
from torch.autograd import Function as Function

from xt.utils import is_main_process

_int_or_tuple_2_t = Union[int, Tuple[int, int]]


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    output_fmt: Format

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)]
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                assert (
                    H == self.img_size[0]
                ), f"Input height ({H}) doesn't match model ({self.img_size[0]})."

                assert (
                    W == self.img_size[1]
                ), f"Input width ({W}) doesn't match model ({self.img_size[1]})."

            else:
                assert (
                    H % self.patch_size[0] == 0
                ), f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                assert (
                    W % self.patch_size[1] == 0
                ), f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


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


# @register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
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


class WindowAttention(nn.Module):
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
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

        # get relative_coords_table
        relative_coords_h = torch.arange(
            -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        )
        relative_coords_w = torch.arange(
            -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        )
        relative_coords_table = (
            torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
            .permute(1, 2, 0)
            .contiguous()
            .unsqueeze(0)
        )  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        else:
            relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
            relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
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
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer(
            "relative_position_index", relative_position_index, persistent=False
        )

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

    def forward(self, x, mask: Optional[torch.Tensor] = None):
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
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(
            -1, self.num_heads
        )
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TwoStreamFusion(nn.Module):
    def __init__(self, mode, dim=None, kernel=3, padding=1):
        """
        A general constructor for neural modules fusing two equal sized tensors
        in forward. Following options are supported:

        "add" / "max" / "min" / "avg"             : respective operations on the two halves.
        "concat"                                  : NOOP.
        "concat_linear_{dim_mult}_{drop_rate}"    : MLP to fuse with hidden dim "dim_mult"
                                                    (optional, def 1.) higher than input dim
                                                    with optional dropout "drop_rate" (def: 0.)
        "ln+concat_linear_{dim_mult}_{drop_rate}" : perform MLP after layernorm on the input.

        """
        super().__init__()
        self.mode = mode
        self.dim = dim
        if mode == "add":
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)).sum(dim=0)
        elif mode == "max":
            self.fuse_fn = (
                lambda x: torch.stack(torch.chunk(x, 2, dim=2)).max(dim=0).values
            )
        elif mode == "min":
            self.fuse_fn = (
                lambda x: torch.stack(torch.chunk(x, 2, dim=2)).min(dim=0).values
            )
        elif mode == "avg":
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)).mean(dim=0)
        elif mode == "concat":
            # x itself is the channel concat version
            self.fuse_fn = lambda x: x
        elif mode == "proj":
            self.fuse_fn = nn.Sequential(
                nn.LayerNorm(dim * 2),
                nn.Linear(dim * 2, dim, bias=False),
            )
        elif "concat_linear" in mode:
            if len(mode.split("_")) == 2:
                dim_mult = 1.0
                drop_rate = 0.0
            elif len(mode.split("_")) == 3:
                dim_mult = float(mode.split("_")[-1])
                drop_rate = 0.0

            elif len(mode.split("_")) == 4:
                dim_mult = float(mode.split("_")[-2])
                drop_rate = float(mode.split("_")[-1])
            else:
                raise NotImplementedError

            if mode.split("+")[0] == "ln":
                self.fuse_fn = nn.Sequential(
                    nn.LayerNorm(dim * 2),
                    Mlp(
                        in_features=dim * 2,
                        hidden_features=int(dim * dim_mult),
                        act_layer=nn.GELU,
                        out_features=dim,
                        drop=drop_rate,
                    ),
                )
            else:
                self.fuse_fn = Mlp(
                    in_features=dim * 2,
                    hidden_features=int(dim * dim_mult),
                    act_layer=nn.GELU,
                    out_features=dim,
                    drop=drop_rate,
                )

        else:
            raise NotImplementedError

        # self.init_weights()

    def init_weights(self):
        "Purely for overriding the proj initialization"
        for name, param in self.fuse_fn.named_parameters():
            if "fc2" in name and "weight" in name:
                with torch.no_grad():
                    torch.diagonal(param.data[:, : self.dim]).add_(0.5)
                    torch.diagonal(param.data[:, self.dim :]).add_(0.5)
            elif "weight" in name:
                nn.init.eye_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        # if "concat_linear" in self.mode:
        #     self.fuse_fn
        #     proj = self.fuse_fn[1]
        #     trunc_normal_(proj.weight, std=0.02)

    def forward(self, x):
        return self.fuse_fn(x)


class ReversibleSwinTransformerV2Block(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            num_heads: Number of attention heads.
            window_size: Window size.
            shift_size: Shift size for SW-MSA.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            pretrained_window_size: Window size in pretraining.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        ws, ss = self._calc_window_shift(window_size, shift_size)
        self.window_size: Tuple[int, int] = ws
        self.shift_size: Tuple[int, int] = ss
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio

        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.seeds = {}

        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            ):
                for w in (
                    slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None),
                ):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def _seed_cuda(self, key):
        """
        Fix seeds to allow for stochastic elements such as
        dropout to be reproduced exactly in activation
        recomputation in the backward pass.

        From RevViT.
        """

        # randomize seeds
        # use cuda generator if available
        if (
            hasattr(torch.cuda, "default_generators")
            and len(torch.cuda.default_generators) > 0
        ):
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)

        self.seeds[key] = seed
        torch.manual_seed(self.seeds[key])

    def _calc_window_shift(
        self, target_window_size, target_shift_size
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        target_window_size = to_2tuple(target_window_size)
        target_shift_size = to_2tuple(target_shift_size)
        window_size = [
            r if r <= w else w
            for r, w in zip(self.input_resolution, target_window_size)
        ]
        shift_size = [
            0 if r <= w else s
            for r, w, s in zip(self.input_resolution, window_size, target_shift_size)
        ]
        return tuple(window_size), tuple(shift_size)

    def _attn(self, x):
        B, H, W, C = x.shape

        # cyclic shift
        has_shift = any(self.shift_size)
        if has_shift:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size[0], -self.shift_size[1]),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_area, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size[0], self.window_size[1], C
        )
        shifted_x = window_reverse(
            attn_windows, self.window_size, self.input_resolution
        )  # B H' W' C

        # reverse cyclic shift
        if has_shift:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
        else:
            x = shifted_x
        return x

    def forward(self, X1, X2):
        # X1, X2: shape is [B, H, W, C]
        assert X1.shape == X2.shape, "Input shapes are different."
        B, H, W, C = X1.shape

        self._seed_cuda("attn")
        attn_out = self.norm1(self._attn(X2))

        self._seed_cuda("droppath1")
        Y1 = X1 + self.drop_path1(attn_out)

        del X1

        Y1 = Y1.reshape(B, -1, C)
        X2 = X2.reshape(B, -1, C)

        self._seed_cuda("mlp")
        mlp_out = self.norm2(self.mlp(Y1))

        self._seed_cuda("droppath2")
        Y2 = X2 + self.drop_path2(mlp_out)

        del X2

        Y1 = Y1.reshape(B, H, W, C)
        Y2 = Y2.reshape(B, H, W, C)

        return Y1, Y2

    def backward_pass(self, Y_1, Y_2, dY_1, dY_2):
        """
        equations for recovering activations:
        X2 = Y2 - MLP(Y1)
        X1 = Y1 - Attn(X2)
        """

        # temporarily record intermediate activation for G
        # and use them for gradient calculcation of G
        with torch.enable_grad():
            Y_1.requires_grad = True

            torch.manual_seed(self.seeds["mlp"])
            g_Y_1 = self.norm2(self.mlp(Y_1))

            torch.manual_seed(self.seeds["droppath2"])
            g_Y_1 = self.drop_path2(g_Y_1)

            g_Y_1.backward(dY_2, retain_graph=True)

        # activation recomputation is by design and not part of
        # the computation graph in forward pass.
        with torch.no_grad():
            X_2 = Y_2 - g_Y_1
            del g_Y_1

            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        # record F activations and calc gradients on F
        with torch.enable_grad():
            X_2.requires_grad = True

            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.norm1(self._attn(X_2))

            torch.manual_seed(self.seeds["droppath1"])
            f_X_2 = self.drop_path1(f_X_2)

            f_X_2.backward(dY_1, retain_graph=True)

        # propagate reverse computed acitvations at the start of
        # the previou block for backprop.s
        with torch.no_grad():
            X_1 = Y_1 - f_X_2

            del f_X_2, Y_1
            dY_2 = dY_2 + X_2.grad

            X_2.grad = None
            X_2 = X_2.detach()

        return X_1, X_2, dY_1, dY_2


class PatchMerging(nn.Module):
    """Patch Merging Layer, modified to support TwoStreamFusion."""

    def __init__(self, dim, out_dim=None, norm_layer=nn.LayerNorm, fusion_module=None):
        """
        Args:
            dim (int): Number of input channels.
            out_dim (int): Number of output channels (or 2 * dim if None)
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
            fusion_module (nn.Module, optional): Fusion module for rev.  Default: None
        """
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)
        self.fusion_module = (
            fusion_module if fusion_module is not None else nn.Identity()
        )

    def forward(self, x):
        # Fuse for revswin first
        x = self.fusion_module(x)
        # Then normal patch merge
        B, H, W, C = x.shape
        assert H % 2 == 0, f"x height ({H}) is not even."
        assert W % 2 == 0, f"x width ({W}) is not even."
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.reduction(x)
        x = self.norm(x)
        return x


class RevBackProp(Function):
    """
    Custom Backpropagation function to allow (A) flushing memory in foward
    and (B) activation recomputation reversibly in backward for gradient calculation.

    Inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(ctx, x, blocks):
        """
        Reversible Forward pass. Any intermediate activations from `buffer_layers` are
        cached in ctx for forward pass. This is not necessary for standard usecases.
        Each reversible layer implements its own forward pass logic.
        """

        X_1, X_2 = torch.chunk(x, 2, dim=-1)
        for _, blk in enumerate(blocks):
            X_1, X_2 = blk(X_1, X_2)

        all_tensors = [X_1.detach(), X_2.detach()]
        ctx.save_for_backward(*all_tensors)
        ctx.blocks = blocks
        # ctx.attn_mask = attn_mask

        return torch.cat([X_1, X_2], dim=-1)

    @staticmethod
    def backward(ctx, dx):
        """
        Reversible Backward pass. Any intermediate activations from `buffer_layers` are
        recovered from ctx. Each layer implements its own logic for backward pass (both
        activation recomputation and grad calculation).
        """
        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve params from ctx for backward
        X_1, X_2 = ctx.saved_tensors
        blocks = ctx.blocks
        # attn_mask = ctx.attn_mask

        for _, blk in enumerate(blocks[::-1]):
            X_1, X_2, dX_1, dX_2 = blk.backward_pass(
                Y_1=X_1,
                Y_2=X_2,
                dY_1=dX_1,
                dY_2=dX_2,
            )

        dx = torch.cat([dX_1, dX_2], dim=-1)

        del dX_1, dX_2, X_1, X_2

        return dx, None, None


class ReversibleSwinTransformerV2Stage(nn.Module):
    """A Swin Transformer V2 Stage."""

    def __init__(
        self,
        dim,
        out_dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        downsample=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
        output_nchw=False,
        use_vanilla_backward=False,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            num_heads: Number of attention heads.
            window_size: Local window size.
            downsample: Use downsample layer at start of the block.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
            pretrained_window_size: Local window size in pretraining.
            output_nchw: Output tensors on NCHW format instead of NHWC.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = (
            tuple(i // 2 for i in input_resolution) if downsample else input_resolution
        )
        self.depth = depth
        self.output_nchw = output_nchw
        self.grad_checkpointing = False

        # patch merging / downsample layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
                fusion_module=TwoStreamFusion(mode="concat_linear_2", dim=dim),
            )
        else:
            assert dim == out_dim
            self.downsample = nn.Identity()

        self.use_vanilla_backward = use_vanilla_backward

        # build blocks
        self.blocks = nn.ModuleList(
            [
                ReversibleSwinTransformerV2Block(
                    dim=out_dim,
                    input_resolution=self.output_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_size,
                )
                for i in range(depth)
            ]
        )

    @staticmethod
    def vanilla_backward(h, blocks):
        """
        Use rev layers without rev backprop, for debugging. Use w/ self.use_vanilla_backward
        """
        # split into hidden states (h) and attention_output (a)
        a, h = torch.chunk(h, 2, dim=-1)
        for blk in blocks:
            a, h = blk(a, h)

        return torch.cat([a, h], dim=-1)

    def forward(self, x):
        x = self.downsample(x)
        x = torch.cat([x, x], dim=-1)

        if self.use_vanilla_backward:
            executing_fn = ReversibleSwinTransformerV2Stage.vanilla_backward
        else:
            executing_fn = RevBackProp.apply

        x = executing_fn(x, self.blocks)

        return x

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class ReversibleSwinTransformerV2(nn.Module):
    """Swin Transformer V2

    A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
        - https://arxiv.org/abs/2111.09883
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "avg",
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_size: _int_or_tuple_2_t = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: Callable = nn.LayerNorm,
        pretrained_window_sizes: Tuple[int, ...] = (0, 0, 0, 0),
        input_dim=2,
        use_vanilla_backward=False,
        upsample=True,
        **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels (of the original model)
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer stage (layer).
            num_heads: Number of attention heads in different layers.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Head dropout rate.
            proj_drop_rate: Projection dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            norm_layer: Normalization layer.
            patch_norm: If True, add normalization after patch embedding.
            pretrained_window_sizes: Pretrained window sizes of each layer.
            output_fmt: Output tensor format if not None, otherwise output 'NHWC' by default.
            upsample: If we use upsample in dense prediction (False for EncDecv2)
        """
        super().__init__()
        if input_dim == in_chans:
            self.input_ada = nn.Identity()
        elif input_dim < in_chans:
            self.input_ada = nn.Conv2d(input_dim, in_chans, 1, 1)
        else:
            raise ValueError(
                "input dim must <= in_chans, otherwise consider change in_chans!"
            )

        self.num_classes = num_classes
        assert global_pool in ("", "avg")
        self.global_pool = global_pool
        self.output_fmt = "NHWC"
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2**self.num_layers)  # x 2 for reversible
        self.feature_info = []

        # ? TODO: what is proper embedding breakdown??
        # ? turns out x2 isn't right since it's the out dim as well.
        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2**i) for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            output_fmt="NHWC",
        )
        self.feature_info += [
            dict(num_chs=embed_dim[0], reduction=2, module="patch_embed")
        ]
        if upsample:
            self.upsample = nn.ModuleList(
                [nn.ConvTranspose2d(embed_dim[0], embed_dim[0], 2, 2)]
            )
        else:
            self.upsample = nn.ModuleList([nn.Identity()])

        dpr = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [
                ReversibleSwinTransformerV2Stage(
                    dim=in_dim,
                    out_dim=out_dim,
                    input_resolution=(
                        self.patch_embed.grid_size[0] // scale,
                        self.patch_embed.grid_size[1] // scale,
                    ),
                    depth=depths[i],
                    downsample=i > 0,
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    pretrained_window_size=pretrained_window_sizes[i],
                    use_vanilla_backward=use_vanilla_backward,
                )
            ]
            in_dim = out_dim
            # scaling = (1 if (i < self.num_layers - 1) else 0)
            if i > 0:
                scale *= 2

            self.upsample.append(nn.Identity())
            self.feature_info += [
                dict(
                    num_chs=out_dim * 2,
                    reduction=4 * scale,
                    module=f"layers.{i}",
                )
            ]

        self.layers = nn.Sequential(*layers)
        # self.norm = norm_layer(self.num_features)
        # self.head = ClassifierHead(
        #     self.num_features,
        #     num_classes,
        #     pool_type=global_pool,
        #     drop_rate=drop_rate,
        #     input_fmt=self.output_fmt,
        # )

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        nod = set()
        for n, m in self.named_modules():
            if any(
                [
                    kw in n
                    for kw in (
                        "cpb_mlp",
                        "logit_scale",
                        "relative_position_bias_table",
                    )
                ]
            ):
                nod.add(n)
        return nod

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^absolute_pos_embed|patch_embed",  # stem and embed
            blocks=r"^layers\.(\d+)"
            if coarse
            else [
                (r"^layers\.(\d+).downsample", (0,)),
                (r"^layers\.(\d+)\.\w+\.(\d+)", None),
                (r"^norm", (99999,)),
            ],
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for layer in self.layers:
            layer.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.input_ada(x)  # * added
        x = self.patch_embed(x)
        outs = [x.permute(0, 3, 1, 2)]
        for layer in self.layers:
            x = layer(x)
            outs.append(x.permute(0, 3, 1, 2))  # * added

        for idx, ox in enumerate(outs[:]):
            outs[idx] = self.upsample[idx](ox)
        # x = self.norm(x) # * removed
        return outs

    # def forward_head(self, x, pre_logits: bool = False):
    #     return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    for k, v in state_dict.items():
        if any([n in k for n in ("relative_position_index", "relative_coords_table")]):
            continue  # skip buffers that should not be persistent
        out_dict[k] = v
    return out_dict


def revswinv2_tiny_window16_256_xview(pretrained=True, **kwargs):
    """ """
    model_args = dict(
        window_size=16,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
    )
    model = ReversibleSwinTransformerV2(**dict(model_args, **kwargs))
    if pretrained:
        if is_main_process():
            print(f"Loading pretrained backbone weights from path {pretrained}...")
        ckpt = torch.load(pretrained, map_location="cpu")
        state_dict = model.state_dict()
        filtered = {}
        unexpected_keys = []
        for k, v in ckpt.items():
            if k in state_dict:
                if state_dict[k].shape != v.shape:
                    if is_main_process():
                        print(f"Skipped {k} for size mismatch")
                        print(state_dict[k].shape, v.shape)
                    continue
                filtered[k] = v
            else:
                unexpected_keys.append(k)
        missing_keys = set(state_dict.keys()) - set(filtered.keys())
        # print("Missing keys: ", missing_keys)
        # print("Unexpected keys: ", unexpected_keys)
        if is_main_process():
            print(model.load_state_dict(filtered, strict=False))
    return model


def revswinv2_base_window16_256_xview(pretrained=True, **kwargs):
    """ """
    model_args = dict(
        window_size=16,
        embed_dim=128,
        depths=(2, 2, 18, 2),
        num_heads=(4, 8, 16, 32),
    )
    model = ReversibleSwinTransformerV2(**dict(model_args, **kwargs))
    if pretrained:
        if is_main_process():
            print(f"Loading pretrained backbone weights from {pretrained}...")
        # ckpt = timm.create_model("swinv2_base_window16_256", pretrained=True).state_dict()
        ckpt = torch.load(pretrained, map_location="cpu")
        state_dict = model.state_dict()
        filtered = {}
        unexpected_keys = []
        for k, v in ckpt.items():
            if k in state_dict:
                if state_dict[k].shape != v.shape:
                    if is_main_process():
                        print(f"Skipped {k} for size mismatch")
                        print(state_dict[k].shape, v.shape)
                    continue
                filtered[k] = v
            else:
                unexpected_keys.append(k)
        missing_keys = set(state_dict.keys()) - set(filtered.keys())
        # print("Missing keys: ", missing_keys)
        # print("Unexpected keys: ", unexpected_keys)
        if is_main_process():
            print(model.load_state_dict(filtered, strict=False))
    return model


def revswinv2_large_window16_256_xview(pretrained=False, **kwargs):
    """ """
    model_args = dict(
        window_size=16,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        pretrained_window_size=(12, 12, 12, 6),
        **kwargs,
    )
    model = ReversibleSwinTransformerV2(**dict(model_args, **kwargs))
    if pretrained:
        print("Loading pretrained backbone weights from path...")
        ckpt = torch.load(pretrained, map_location="cpu")
        state_dict = model.state_dict()
        filtered = {}
        unexpected_keys = []
        for k, v in ckpt.items():
            if k in state_dict:
                if state_dict[k].shape != v.shape:
                    print(f"Skipped {k} for size mismatch")
                    print(state_dict[k].shape, v.shape)
                    continue
                filtered[k] = v
            else:
                unexpected_keys.append(k)
        missing_keys = set(state_dict.keys()) - set(filtered.keys())
        print("Missing keys: ", missing_keys)
        # print("Unexpected keys: ", unexpected_keys)
        msg = model.load_state_dict(filtered, strict=False)
        print(msg)
    return model