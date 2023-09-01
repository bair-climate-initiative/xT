# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/mmseg/models/backbones/swin_transformer.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.autograd import Function as Function

# # Using fused window kernels
# try:
#     import os, sys

#     kernel_path = os.path.abspath(os.path.join('..'))
#     sys.path.append(kernel_path)
#     from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

# except:
WindowProcess = None
WindowProcessReverse = None
print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

class Mlp(nn.Module):
    """Multilayer perceptron."""

    def __init__(
        self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
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
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)).sum(
                dim=0
            )
        elif mode == "max":
            self.fuse_fn = (
                lambda x: torch.stack(torch.chunk(x, 2, dim=2))
                .max(dim=0)
                .values
            )
        elif mode == "min":
            self.fuse_fn = (
                lambda x: torch.stack(torch.chunk(x, 2, dim=2))
                .min(dim=0)
                .values
            )
        elif mode == "avg":
            self.fuse_fn = lambda x: torch.stack(torch.chunk(x, 2, dim=2)).mean(
                dim=0
            )
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
                        drop_rate=drop_rate,
                    ),
                )
            else:
                self.fuse_fn = Mlp(
                    in_features=dim * 2,
                    hidden_features=int(dim * dim_mult),
                    act_layer=nn.GELU,
                    out_features=dim,
                    drop_rate=drop_rate,
                )

        else:
            raise NotImplementedError
    
    def init_weights(self):
        "Purely for overriding the proj initialization"
        if "proj" in self.mode:
            proj = self.fuse_fn[1]
            trunc_normal_(proj.weight, std=0.02)
            # nn.init.eye_(proj.weight[:, :self.dim])
            # nn.init.eye_(proj.weight[:, self.dim:])
            with torch.no_grad():
                torch.diagonal(proj.weight.data[:, :self.dim]).add_(0.5)
                torch.diagonal(proj.weight.data[:, self.dim:]).add_(0.5)
                # proj.weight += torch.randn(*proj.weight.shape)*0.02
                # proj.weight /= 2 # normalize outgoing variance

    def forward(self, x):
        return self.fuse_fn(x)

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1)
                        * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] *
            self.window_size[1], -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ReversibleSwinTransformerBlock(nn.Module):
    """Reversible Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        fused_window_process=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop
        )

        self.H = None
        self.W = None

        self.seeds = {}
        self.fused_window_process = fused_window_process

    def seed_cuda(self, key):
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

    def forward_attn(self, x, mask_matrix):
        """ Forward function for windowed attention. No drop_path.
         Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.       
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            attn_mask = mask_matrix
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            attn_mask = None
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        return x

    def forward_mlp(self, x):
        """ Forward function for mlp. """
        return self.mlp(self.norm2(x))

    def forward(self, X1, X2, mask_matrix):
        """Reversible forward function with rewiring.

        Y_1 = X_1 + Attn(X_2)
        Y_2 = X_2 + MLP(Y_1)

        Args:
            x1, x2: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        assert X1.shape == X2.shape, "Input shapes are different."

        self.seed_cuda("attn")
        attn_out = self.forward_attn(X2, mask_matrix)

        self.seed_cuda("droppath")
        Y1 = X1 + self.drop_path(attn_out)

        # Free memory
        del X1

        self.seed_cuda("mlp")
        mlp_out = self.forward_mlp(Y1)

        torch.manual_seed(self.seeds["droppath"])
        Y2 = X2 + self.drop_path(mlp_out)

        del X2

        return Y1, Y2

    def backward_pass(self, Y_1, Y_2, dY_1, dY_2, mask_matrix):
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
            g_Y_1 = self.forward_mlp(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = self.drop_path(g_Y_1)

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
            f_X_2 = self.forward_attn(X_2, mask_matrix)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = self.drop_path(f_X_2)

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

    def backward_pass_recover(self, Y_1, Y_2, mask_matrix):
        """
        Use equations to recover activations and return them.
        Used for streaming the backward pass.
        """
        with torch.enable_grad():
            Y_1.requires_grad = True

            torch.manual_seed(self.seeds["mlp"])
            g_Y_1 = self.forward_mlp(Y_1)

            torch.manual_seed(self.seeds["droppath"])
            g_Y_1 = self.drop_path(g_Y_1)

        with torch.no_grad():
            X_2 = Y_2 - g_Y_1

        with torch.enable_grad():
            X_2.requires_grad = True

            torch.manual_seed(self.seeds["attn"])
            f_X_2 = self.forward_attn(X_2, mask_matrix)

            torch.manual_seed(self.seeds["droppath"])
            f_X_2 = self.drop_path(f_X_2)

        with torch.no_grad():
            X_1 = Y_1 - f_X_2

        # Keep tensors around to do backprop on the graph.
        ctx = [X_1, X_2, Y_1, g_Y_1, f_X_2]
        return ctx

    def backward_pass_grads(self, X_1, X_2, Y_1, g_Y_1, f_X_2, dY_1, dY_2):
        """
        Receive intermediate activations and inputs to backprop through.
        """

        with torch.enable_grad():
            g_Y_1.backward(dY_2)

        with torch.no_grad():
            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        with torch.enable_grad():
            f_X_2.backward(dY_1)

        with torch.no_grad():
            dY_2 = dY_2 + X_2.grad
            X_2.grad = None
            X_2.detach()

        return dY_1, dY_2


class PatchMerging(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class RevBackProp(Function):
    """
    Custom Backpropagation function to allow (A) flushing memory in foward
    and (B) activation recomputation reversibly in backward for gradient calculation.

    Inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
    """

    @staticmethod
    def forward(ctx, x, blocks, buffer):
        """
        Reversible Forward pass. Any intermediate activations from `buffer_layers` are
        cached in ctx for forward pass. This is not necessary for standard usecases.
        Each reversible layer implements its own forward pass logic.
        """

        H, W, attn_mask, _ = buffer

        X_1, X_2 = torch.chunk(x, 2, dim=-1)
        for _, blk in enumerate(blocks):
            blk.H, blk.W = H, W
            X_1, X_2 = blk(X_1, X_2, attn_mask)

        all_tensors = [X_1.detach(), X_2.detach()]
        ctx.save_for_backward(*all_tensors)
        ctx.blocks = blocks
        ctx.attn_mask = attn_mask

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
        attn_mask = ctx.attn_mask

        for _, blk in enumerate(blocks[::-1]):

            X_1, X_2, dX_1, dX_2 = blk.backward_pass(
                Y_1=X_1,
                Y_2=X_2,
                dY_1=dX_1,
                dY_2=dX_2,
                mask_matrix=attn_mask,
            )

        dx = torch.cat([dX_1, dX_2], dim=-1)

        del dX_1, dX_2, X_1, X_2

        return dx, None, None

class RevBackPropFast(RevBackProp):
    @staticmethod
    def backward(ctx, dx):
        """Overwrite backward by using PyTorch Streams to parallelize."""

        dX_1, dX_2 = torch.chunk(dx, 2, dim=-1)

        # retrieve params from ctx for backward
        X_1, X_2, *int_tensors = ctx.saved_tensors

        layers = ctx.blocks
        attn_mask = ctx.attn_mask

        # Keep a dictionary of events to synchronize on
        # Each is for the completion of a recalculation (f) or gradient calculation (b)
        events = {}
        for i in range(len(layers)):
            events[f"f{i}"] = torch.cuda.Event()
            events[f"b{i}"] = torch.cuda.Event()

        # Run backward staggered on two streams, which were defined globally in __init__
        # Initial pass
        with torch.cuda.stream(s1):
            layer = layers[-1]
            prev = layer.backward_pass_recover(
                Y_1=X_1,
                Y_2=X_2,
                mask_matrix=attn_mask
            )

            events["f0"].record(s1)

        # Stagger streams based on iteration
        for i, (this_layer, next_layer) in enumerate(
            zip(layers[1:][::-1], layers[:-1][::-1])
        ):
            if i % 2 == 0:
                stream1 = s1
                stream2 = s2
            else:
                stream1 = s2
                stream2 = s1

            with torch.cuda.stream(stream1):
                # b{i} waits on b{i-1}
                if i > 0:
                    events[f"b{i-1}"].synchronize()

                if i % 2 == 0:
                    dY_1, dY_2 = this_layer.backward_pass_grads(*prev, dX_1, dX_2)
                else:
                    dX_1, dX_2 = this_layer.backward_pass_grads(*prev, dY_1, dY_2)

                events[f"b{i}"].record(stream1)

            with torch.cuda.stream(stream2):
                # f{i} waits on f{i-1}
                events[f"f{i}"].synchronize()

                prev = next_layer.backward_pass_recover(
                    Y_1=prev[0],
                    Y_2=prev[1],
                    mask_matrix=attn_mask
                )

                events[f"f{i+1}"].record(stream2)

        # Last iteration
        if len(layers) - 1 % 2 == 0:
            stream2 = s1
        else:
            stream2 = s2
        next_layer = layers[0]
            
        with torch.cuda.stream(stream2):
            # stream2.wait_event(events[f"b{len(layers)-2}_end"])
            events[f"b{len(layers)-2}"].synchronize()
            if len(layers) - 1 % 2 == 0:
                dY_1, dY_2 = next_layer.backward_pass_grads(*prev, dX_1, dX_2)
                dx = torch.cat([dY_1, dY_2], dim=-1)
            else:
                dX_1, dX_2 = next_layer.backward_pass_grads(*prev, dY_1, dY_2)
                dx = torch.cat([dX_1, dX_2], dim=-1)
            events[f"b{len(layers)-1}"].record(stream2)

        # Synchronize, for PyTorch 1.9
        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)
        events[f"b{len(layers)-1}"].synchronize()

        del int_tensors
        del dX_1, dX_2, dY_1, dY_2, X_1, X_2, prev[:]
        return dx, None, None


class ReversibleLayer(nn.Module):
    """A Reversible Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        fused_window_process=False,
        lateral_fusion="avg",
        two_stream_input=True,
        fast_backprop=False
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                ReversibleSwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(
                        drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    fused_window_process=fused_window_process,
                )
                for i in range(depth)
            ]
        )


        # patch merging layer
        if downsample is not None:
            # lateral fusion strategy: input [B, L, 2*C] -> [B, L, C]
            self.lateral_fuse = TwoStreamFusion(lateral_fusion, dim=dim)
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # keep track if inputs are two-stream already
        self.two_stream_input = two_stream_input

        self.use_vanilla_backward = False
        self.use_fast_backprop = fast_backprop

    @staticmethod
    def vanilla_backward(h, blocks, buffer):
        """
        Use rev layers without rev backprop, for debugging. Use w/ self.use_vanilla_backward 
        """
        # split into hidden states (h) and attention_output (a)
        H, W, attn_mask, use_checkpoint = buffer

        # ? Why swapped?
        h, a = torch.chunk(h, 2, dim=-1)
        for _, blk in enumerate(blocks):
            blk.H, blk.W = H, W
            if use_checkpoint:
                a, h = checkpoint.checkpoint(blk, a, h, attn_mask)
            else:
                a, h = blk(a, h, attn_mask)

        return torch.cat([a, h], dim=-1)  # fuse by averaging

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1,
                                         self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        if not self.two_stream_input: 
            x = torch.cat([x, x], dim=-1)

        if self.use_vanilla_backward:
            executing_fn = ReversibleLayer.vanilla_backward
        elif self.use_fast_backprop:
            executing_fn = RevBackPropFast.apply
        else:
            executing_fn = RevBackProp.apply

        x = executing_fn(
            x, self.blocks, (H, W, attn_mask, self.use_checkpoint))

        # for blk in self.blocks:
        #     blk.H, blk.W = H, W
        #     if self.use_checkpoint:
        #         x = checkpoint.checkpoint(blk, x, attn_mask)
        #     else:
        #         x = blk(x, attn_mask)

        if self.downsample is not None:
            # Fuses [B, L, 2C] (from 2 streams) to [B, L, C]
            x = self.lateral_fuse(x)
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x_down, Wh, Ww
            # return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W
            # return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(
                x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class ReversibleSwinTransformer(nn.Module):
    """Reversible Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        input_dim=3,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        fused_window_process=False,
        lateral_fusion="avg",
        fast_backprop=False,
        **kwargs
    ):
        super().__init__()

        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # self.out_indices = out_indices
        # self.frozen_stages = frozen_stages

        # Channel matching
        if input_dim == in_chans:
            self.input_ada = nn.Identity()
        elif input_dim < in_chans:
            self.input_ada = nn.Conv2d(input_dim,in_chans,1,1)
        else:
            raise ValueError("input dim must <= in_chans, otherwise consider change in_chans!")

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            img_size = to_2tuple(img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                img_size[0] // patch_size[0],
                img_size[1] // patch_size[1],
            ]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(
                    1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # Initialize global stream for device
        if fast_backprop:
            global s1, s2 
            s1 = torch.cuda.Stream(device=torch.cuda.current_device())
            s2 = torch.cuda.Stream(device=torch.cuda.current_device())

        # build layers
        self.feature_info = []
        self.layers = nn.ModuleList()
        self.feature_info += [dict(num_chs=embed_dim, reduction=2, module=f'patch_embed')]
        self.upsample = nn.ModuleList()
        self.upsample.append(
            nn.ConvTranspose2d(embed_dim,embed_dim,2,2)
        )
        for i_layer in range(self.num_layers):
            layer = ReversibleLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (
                    i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                fused_window_process=fused_window_process,
                lateral_fusion=lateral_fusion,
                two_stream_input=False,
                fast_backprop=fast_backprop,
            )
            self.layers.append(layer)
            scaling = (1 if (i_layer < self.num_layers - 1) else 0)
            # num_chs = int(embed_dim * 2 ** (i_layer+scaling ))
            # always + 1 since we have 2x channels in rev at the end
            num_chs = int(embed_dim * 2 ** (i_layer + 1))
            if scaling:
                self.upsample.append(nn.ConvTranspose2d(num_chs,num_chs,2,2))
            else:
                self.upsample.append(nn.Identity())
            self.feature_info += [dict(num_chs=num_chs, reduction=4 * 2 ** i_layer, module=f'layers.{i_layer}')]

        # num_features = [int(embed_dim * 2 ** i)
        #                 for i in range(self.num_layers)]
        # self.num_features = num_features
        # if not rev, then self.num_features = [int(embed_dim * 2 ** (self.num_layers - 1))] 
        # but rev has 2x channels, so features is * 2 as last layer isn't downsampled
        self.num_features = int(embed_dim * 2 ** self.num_layers)
        self.mlp_ratio = mlp_ratio

        # add a norm layer for each output
        # for i_layer in out_indices:
        #     layer = norm_layer(num_features[i_layer])
        #     layer_name = f"norm{i_layer}"
        #     self.add_module(layer_name, layer)
        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for layer in self.layers:
            if layer.downsample is not None:
                layer.lateral_fuse.init_weights() 
        # self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.
    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.02)
            nn.init.constant_(m.weight, 1.0)

    def reshape(self, x):
        _, hw, d = x.shape
        r = int(np.sqrt(hw))
        x_reshaped = x.view(-1,r,r,d).permute(0,3,1,2).contiguous()
        return x_reshaped

    def forward_features(self, x):
        """Forward function."""
        #print(x.shape)
        x = self.input_ada(x)
        x = self.patch_embed(x)

        # print("start outs")
        # print(x.shape)

        # No need for reshape, processed as [B x C x H x W]
        outs = [x]
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic"
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            x, Wh, Ww = layer(x, Wh, Ww)

            #print(x.shape)
            outs.append(self.reshape(x))
            # if i in self.out_indices:
            #     norm_layer = getattr(self, f"norm{i}")
            #     x_out = norm_layer(x_out)

            #     out = x_out.view(-1, H, W,
            #                      self.num_features[i]).permute(0, 3, 1, 2).contiguous()
            #     outs["res{}".format(i + 2)] = out

        #print("out shapes after upsamples")
        for idx, ox in enumerate(outs[:]):
            outs[idx] = self.upsample[idx](ox)
            #print(outs[idx].shape)
        
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x
        # apply norm to chunks then re-concat
        # B L 2C -> (B L C, B L C) -> # B L 2C
        # ? for true Norm -> Cat
        # x = torch.cat([self.norm(chunk) for chunk in x.chunk(2, dim=2)], dim=2)
        # Basically doing Cat -> Norm

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        x = self.head(x)
        return x

# Trained on patch4_window7_224
def revswinv1_tiny_window7_224_xview(pretrained=False, **kwargs):
    """
    """
    model_kwargs = dict(
        window_size=16, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), **kwargs)
    model =  ReversibleSwinTransformer(**model_kwargs)
    if pretrained:
        ckpt = torch.load(pretrained,map_location='cpu')
        state_dict = model.state_dict()
        filtered = {}
        for k,v in ckpt.items():
            if k in state_dict and state_dict[k].shape != v.shape:
                print(f"Skipped {k} for size mismatch")
                continue
            filtered[k]=v
        model.load_state_dict(filtered,strict=False)
    return model

REVSWIN_CFG = dict(
    revswinv1_tiny_window7_224_xview=revswinv1_tiny_window7_224_xview,
)