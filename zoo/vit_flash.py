# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer 
from flash_attn.modules.mha import MHA as FlashMHA
from flash_attn.modules.mlp import Mlp as FlashMlp
from einops import rearrange
from timm.models.layers import DropPath, Mlp
from timm.models.vision_transformer import Attention, LayerScale
from .utils import interpolate_pos_embed



def load_checkpoint(model, pretrained: str):
    checkpoint_model = torch.load(pretrained, map_location='cpu')

    print(f"Load pre-trained checkpoint from: {pretrained}")
    # checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    if 'pos_embed' in checkpoint_model and checkpoint_model['pos_embed'].shape != state_dict['pos_embed'].shape:
        print(f"Adjusting class token for pos_embed in pretrained checkpoint")
        checkpoint_model['pos_embed'] = checkpoint_model['pos_embed'][:, 1:, :]

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    return model


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            flash_attn=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = FlashMHA(dim, num_heads=num_heads, qkv_proj_bias=qkv_bias, dropout=attn_drop, use_flash_attn=flash_attn)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(global_pool='', 
                         class_token=False, 
                         num_classes=0, 
                         qkv_bias=True,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6),
                         block_fn=partial(Block, flash_attn=True),
                         *args, **kwargs)

    def forward(self, x):
        x = self.forward_features(x) 
        return x


def vit_base_patch16(pretrained = False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        **kwargs)
    del model.fc_norm
    model.forward = model.forward_features
    if pretrained:
        model = load_checkpoint(model, pretrained)
    return model


def vit_large_patch16(pretrained = False, channels_last=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained)
    return model


def vit_huge_patch14(pretrained = False, channels_last=True, **kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        **kwargs)
    if pretrained:
        model = load_checkpoint(model, pretrained)
    return model

registry = {
    "vit_base_patch16": vit_base_patch16,
    "vit_large_patch16": vit_large_patch16,
    "vit_huge_patch14": vit_huge_patch14,
}