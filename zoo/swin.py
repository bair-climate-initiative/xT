import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.fx_features import register_notrace_function
from timm.models.helpers import build_model_with_cfg, named_apply
from timm.models.layers import (
    DropPath,
    Mlp,
    PatchEmbed,
    _assert,
    to_2tuple,
    to_ntuple,
    trunc_normal_,
)
from timm.models.registry import register_model
from timm.models.swin_transformer_v2 import BasicLayer, PatchMerging


class SwinTransformerV2Xview(nn.Module):
    r"""Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`
            - https://arxiv.org/abs/2111.09883
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        pretrained_window_sizes=(0, 0, 0, 0),
        input_dim=3,
        **kwargs,
    ):
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
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches

        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.feature_info = []
        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.upsample.append(nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2))
        self.feature_info += [
            dict(num_chs=embed_dim, reduction=2, module=f"patch_embed")
        ]
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    self.patch_embed.grid_size[0] // (2**i_layer),
                    self.patch_embed.grid_size[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                pretrained_window_size=pretrained_window_sizes[i_layer],
            )
            self.layers.append(layer)
            scaling = 1 if (i_layer < self.num_layers - 1) else 0
            num_chs = int(embed_dim * 2 ** (i_layer + scaling))
            if scaling:
                self.upsample.append(nn.ConvTranspose2d(num_chs, num_chs, 2, 2))
            else:
                self.upsample.append(nn.Identity())
            self.feature_info += [
                dict(
                    num_chs=num_chs,
                    reduction=4 * 2**i_layer,
                    module=f"layers.{i_layer}",
                )
            ]

        # self.norm = norm_layer(self.num_features)

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
        nod = {"absolute_pos_embed"}
        for n, m in self.named_modules():
            if any(
                [
                    kw in n
                    for kw in ("cpb_mlp", "logit_scale", "relative_position_bias_table")
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
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg")
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def reshape(self, x):
        _, hw, d = x.shape
        r = int(np.sqrt(hw))
        x_reshaped = x.view(-1, r, r, d).permute(0, 3, 1, 2)
        return x_reshaped

    def forward_features(self, x):
        x = self.input_ada(x)
        x = self.patch_embed(x)
        outs = [self.reshape(x)]
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
            outs.append(self.reshape(x))
        for idx, ox in enumerate(outs[:]):
            outs[idx] = self.upsample[idx](ox)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x


def swinv2_tiny_window16_256_xview(pretrained=False, **kwargs):
    """ """
    model_kwargs = dict(
        window_size=16,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        **kwargs,
    )
    model = SwinTransformerV2Xview(**model_kwargs)
    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu")
        state_dict = model.state_dict()
        filtered = {}
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape != v.shape:
                print(f"Skipped {k} for size mismatch")
                continue
            filtered[k] = v
        model.load_state_dict(filtered, strict=False)
    return model


def swinv2_large_window12_192_xview(pretrained=False, **kwargs):
    """ """
    model_kwargs = dict(
        window_size=12,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        **kwargs,
    )
    model = SwinTransformerV2Xview(**model_kwargs)
    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu")
        state_dict = model.state_dict()
        filtered = {}
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape != v.shape:
                print(f"SKipped {k} for size mismatch")
                continue
            filtered[k] = v
        model.load_state_dict(filtered, strict=False)
    return model


def swinv2_large_window16_256_xview(pretrained=False, **kwargs):
    """ """
    model_kwargs = dict(
        window_size=16,
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6),
        **kwargs,
    )
    model = SwinTransformerV2Xview(**model_kwargs)
    if pretrained:
        ckpt = torch.load(pretrained, map_location="cpu")
        state_dict = model.state_dict()
        filtered = {}
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape != v.shape:
                print(f"SKipped {k} for size mismatch")
                continue
            filtered[k] = v
        model.load_state_dict(filtered, strict=False)
    return model


SWIN_CFG = dict(
    swinv2_tiny_window16_256_xview=swinv2_tiny_window16_256_xview,
    swinv2_large_window12_192_xview=swinv2_large_window12_192_xview,
    swinv2_large_window16_256_xview=swinv2_large_window16_256_xview,
)
