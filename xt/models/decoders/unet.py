import math
from functools import lru_cache
from typing import Any

import numpy as np
import timm
import torch
import torch.hub
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.nn import Dropout2d

from hyp.attention.hyper_attn import HyperAttention

from ..context_encoders.transformer_xl import (ContextEncoderConfig,
                                               MemTransformerLM)
from .pos_embed import get_2d_sincos_pos_embed as get_2d_sincos_pos_embed_base
from .vit import MAEDecoder
from .vit import registry as VIT_CFG

default_decoder_filters = [48, 96, 176, 256]
default_last = 48


@lru_cache(maxsize=32)
def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    i=0,
    j=0,
):
    return get_2d_sincos_pos_embed_base(embed_dim, grid_size, cls_token, i, j)


def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data = nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @property
    def first_layer_params_names(self):
        return ["conv1.conv"]


class BasicConvAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        dilation=1,
        activation=nn.SiLU,
        bias=True,
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.use_act = activation is not None
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class Conv1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=dilation,
            activation=None,
            bias=bias,
        )


class Conv3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            activation=None,
        )


class ConvSilu1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=dilation,
            activation=nn.SiLU,
        )


class ConvSilu3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilation,
            activation=nn.SiLU,
        )


class BasicUpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        activation=nn.SiLU,
        mode="nearest",
    ):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * 1
        self.op = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=1,
        )
        self.use_act = activation is not None
        self.mode = mode
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode=self.mode)
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class UNetDecoder(nn.Module):
    def __init__(
        self,
        bottleneck_type,
        filters,
        decoder_block,
        decoder_filters,
        skip_decoder,
        last_upsample_filters,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.decoder_filters = decoder_filters
        self.last_upsample_filters = last_upsample_filters
        self.filters = filters
        self.bottleneck_type = bottleneck_type
        self.decoder_block = decoder_block
        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))]
        )
        decoder_cls = UnetDecoderLastConv
        if skip_decoder:
            decoder_cls = UnetDecoderLastConvSkip
        self.vessel_mask = decoder_cls(
            self.decoder_filters[0], self.last_upsample_filters, 1
        )
        self.fishing_mask = decoder_cls(
            self.decoder_filters[0], self.last_upsample_filters, 1
        )
        self.center_mask = decoder_cls(
            self.decoder_filters[0], self.last_upsample_filters, 1
        )
        self.length_mask = decoder_cls(
            self.decoder_filters[0], self.last_upsample_filters, 1
        )

    def forward(self, enc_results, x_skip):
        x = enc_results[-1]
        bottlenecks = self.bottlenecks
        pp = []
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = -(idx + 1)
            x = self.decoder_stages[rev_idx](x)
            a = x.shape
            x = bottleneck(x, enc_results[rev_idx - 1])
            b = x.shape
            pp.append((a, b))

        fishing_mask = self.fishing_mask(x, x_skip).contiguous(
            memory_format=torch.contiguous_format
        )
        vessel_mask = self.vessel_mask(x, x_skip).contiguous(
            memory_format=torch.contiguous_format
        )
        center_mask = self.center_mask(x, x_skip).contiguous(
            memory_format=torch.contiguous_format
        )
        length_mask = self.length_mask(x, x_skip).contiguous(
            memory_format=torch.contiguous_format
        )
        output = {
            "fishing_mask": fishing_mask,
            "vessel_mask": vessel_mask,
            "center_mask": center_mask,
            "length_mask": length_mask,
        }
        return output

    def get_decoder(self, layer):
        in_channels = (
            self.filters[layer + 1]
            if layer + 1 == len(self.decoder_filters)
            else self.decoder_filters[layer + 1]
        )
        return self.decoder_block(
            in_channels,
            self.decoder_filters[layer],
            self.decoder_filters[max(layer, 0)],
        )


class SemanticSegDecoder(nn.Module):
    def __init__(
        self,
        bottleneck_type,
        filters,
        decoder_block,
        decoder_filters,
        skip_decoder,
        last_upsample_filters,
        num_classes,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.decoder_filters = decoder_filters
        self.last_upsample_filters = last_upsample_filters
        self.filters = filters
        self.bottleneck_type = bottleneck_type
        self.decoder_block = decoder_block
        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )
        self.decoder_stages = nn.ModuleList(
            [self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))]
        )
        decoder_cls = UnetDecoderLastConv
        if skip_decoder:
            decoder_cls = UnetDecoderLastConvSkip
        self.seg = decoder_cls(
            self.decoder_filters[0], self.last_upsample_filters, num_classes
        )

    def forward(self, enc_results, x_skip):
        x = enc_results[-1]
        bottlenecks = self.bottlenecks
        pp = []
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = -(idx + 1)
            x = self.decoder_stages[rev_idx](x)
            a = x.shape
            x = bottleneck(x, enc_results[rev_idx - 1])
            b = x.shape
            pp.append((a, b))

        seg = self.seg(x, x_skip).contiguous(memory_format=torch.contiguous_format)
        output = {
            "seg": seg,
        }
        return output

    def get_decoder(self, layer):
        in_channels = (
            self.filters[layer + 1]
            if layer + 1 == len(self.decoder_filters)
            else self.decoder_filters[layer + 1]
        )
        return self.decoder_block(
            in_channels,
            self.decoder_filters[layer],
            self.decoder_filters[max(layer, 0)],
        )


class TransformerXLContextModel(nn.Module):
    def __init__(
        self, xl_config, crop_size, reduction, d_model, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        transformer_xl_config = {
            "no_memory": xl_config.no_memory,
            "n_layer": xl_config.n_layer,
        }

        n_length = (crop_size // reduction) ** 2
        n_crops = 9
        n_token = d_model
        cutoffs = [n_token // 2]
        tie_projs = [False] + [True] * len(cutoffs)
        xl_args = dict(
            n_token=n_token,
            n_layer=4,
            n_head=2,
            d_model=d_model,
            d_head=2,
            d_inner=d_model,
            dropout=0.1,
            dropatt=0.1,
            tie_weight=False,
            d_embed=d_model,
            div_val=1,
            tie_projs=tie_projs,
            pre_lnorm=True,
            tgt_len=n_length,
            ext_len=n_length,
            mem_len=n_length * 4,
            cutoffs=cutoffs,
            attn_type=0,
        )

        xl_args.update(transformer_xl_config)
        self.model = MemTransformerLM(**xl_args)
        self.classification_mode = xl_config.get("classification_mode")
        if self.classification_mode:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.init_weights()

    def init_weights(self, pretrained=None):
        if self.classification_mode:
            nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, enc_results, mem, skip=False):
        xx = enc_results[-1]  # N C H W
        old_shape = xx.shape
        xx = rearrange(xx, "n c h w -> (h w) n c")
        L = xx.shape[0]
        N = xx.shape[1]
        if self.classification_mode:
            xx = torch.cat([xx, self.cls_token.repeat(1, N, 1)], dim=0)  # L+1 N C
        xx = self.model(xx, xx, *mem)
        pred_out, mem = xx[0], xx[1:]
        pred_out = pred_out.permute(1, 2, 0)  # N C L+1

        if self.classification_mode:
            assert pred_out.shape[0] == old_shape[0]
            assert pred_out.shape[2] == L + 1
            enc_results[-1] = pred_out[:, :, -1][..., None, None]
        else:
            pred_out = pred_out.view(*old_shape)
            assert pred_out.shape[0] == old_shape[0]
            assert pred_out.shape[1] == old_shape[1]
            if skip:
                enc_results[-1] = torch.cat(
                    [enc_results[-1], pred_out], dim=1
                )  # overwrite
            else:
                enc_results[-1] = pred_out  # overwrite
        mem = mem
        return enc_results, mem


class LLMTransformerXLContextModel(nn.Module):
    def __init__(
        self, xl_config, crop_size, reduction, d_model, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        transformer_xl_config = {
            "no_memory": xl_config.no_memory,
            "n_layer": xl_config.n_layer,
        }
        hidden_size = xl_config.hidden_size
        mlp_ratio = xl_config.mlp_ratio
        num_heads = xl_config.mlp_ratio

        n_length = (crop_size // reduction) ** 2
        n_crops = 9
        n_token = d_model
        cutoffs = [n_token // 2]
        tie_projs = [False] + [True] * len(cutoffs)
        xl_args = dict(
            n_token=n_token,
            n_layer=4,
            n_head=2,
            d_model=d_model,
            d_head=2,
            d_inner=d_model,
            dropout=0.1,
            dropatt=0.1,
            tie_weight=False,
            d_embed=d_model,
            div_val=1,
            tie_projs=tie_projs,
            pre_lnorm=True,
            tgt_len=n_length,
            ext_len=n_length,
            mem_len=n_length * 4,
            cutoffs=cutoffs,
            attn_type=0,
        )

        xl_args.update(transformer_xl_config)
        # self.model = MemTransformerLM(**xl_args)
        self.input_proj = nn.Linear(d_model, hidden_size)
        self.out_proj = nn.Linear(hidden_size, d_model)
        self.layers = nn.Sequential(
            *[
                LLMLayer(
                    hidden_size,
                    hidden_size * mlp_ratio,
                    num_heads,
                    causal=False,
                    attention_method=xl_config.attention_method,
                )
                for _ in range(xl_config.n_layer)
            ]
        )
        self.hidden_size = hidden_size
        self.classification_mode = xl_config.get("classification_mode")
        if self.classification_mode:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mem_chip = xl_config.mem_chip
        self.init_weights()

    def init_weights(self, pretrained=None):
        if self.classification_mode:
            nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, enc_results, mem, skip=False, cord=(0, 0)):
        xx = enc_results[-1]  # N C H W
        N, C, H, W = xx.shape
        xx = rearrange(xx, "n c h w -> n (h w) c")
        raw_xx = xx
        xx = self.input_proj(xx)
        i, j = cord
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size, H, cls_token=False, i=i, j=j
        )
        xx = xx + torch.tensor(pos_embed).to(xx)
        _, L, D = xx.shape
        # L = xx.shape[0]
        # N = xx.shape[1]

        if self.classification_mode:
            xx = torch.cat([xx, self.cls_token.repeat(N, 1, 1)], dim=1)  #  N L+1 C
        base_len = xx.shape[1]
        keep_len = base_len * self.mem_chip
        new_mem = []
        for idx, layer in enumerate(self.layers):
            if not mem:
                xx_i = xx
            else:
                xx_i = torch.cat([mem[idx], xx], dim=1)[:, -keep_len:]
            new_mem.append(xx_i.detach())
            xx = layer(xx_i)[:, -base_len:]  # N L D

        # pred_out, mem = xx[0], xx[1:]
        xx = self.out_proj(xx) + raw_xx
        pred_out = xx.permute(0, 2, 1)  # N C L+1

        if self.classification_mode:
            assert pred_out.shape == (N, C, L + 1)
            enc_results[-1] = pred_out[:, :, -1][..., None, None]
        else:
            assert pred_out.shape == (N, C, H * W)
            pred_out = pred_out.view(N, C, H, W)
            if skip:
                enc_results[-1] = torch.cat(
                    [enc_results[-1], pred_out], dim=1
                )  # overwrite
            else:
                enc_results[-1] = pred_out  # overwrite
        mem = mem
        return enc_results, mem


class ClassificationDecoder(nn.Module):
    def __init__(self, in_dim, num_classes, mlp_ratio=4):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim * mlp_ratio),
            nn.GELU(),
            nn.LayerNorm(in_dim * mlp_ratio),
            nn.Linear(in_dim * mlp_ratio, num_classes),
        )

    def forward(self, enc_results, mem):
        xx = enc_results[-1]  # N C H W
        xx = xx.mean((-1, -2))  # GAP->N X C
        logits = self.layers(xx)  # CLS
        return {"label": logits}


class LLMAttention(nn.Module):
    def __init__(
        self,
        dim,
        inner_dim,
        num_heads,
        causal=False,
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.proj = nn.Linear(inner_dim, dim)
        assert inner_dim % num_heads == 0, (inner_dim, num_heads)
        self.num_heads = num_heads
        self.attn = HyperAttention(
            input_dim=inner_dim // num_heads,
            lsh_num_projs=7,
            block_size=256,
            sample_size=256,
            min_seq_len=4096,
        )
        self.causal = causal
        # self.query_key_value = torch.nn.Linear(
        #     hidden_size,
        #     3 * self.inner_hidden_size,
        #     bias=bias,
        #     dtype=params_dtype,
        # )

    def forward(self, x):
        """
        X: N L H
        """
        B, L, D = x.shape
        q, k, v = (
            self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )  # B H L D // num_heads
        attn_out = self.attn(q, k, v, causal=self.causal).permute(
            0, 2, 1, 3
        )  # B H L D // num_heads
        attn_out = attn_out.reshape(B, L, -1).contiguous()
        attn_out = self.proj(attn_out)

        return attn_out


class ViTAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_mask=False,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_mask = use_mask
        if use_mask:
            self.att_mask = nn.Parameter(torch.Tensor(self.num_heads, 196, 196))

    def forward(self, x):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_mask:
            attn = attn * torch.sigmoid(self.att_mask).expand(B, -1, -1, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


from transformers.models.llama.modeling_llama import LlamaRMSNorm


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.pretraining_tp = 1
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LLMLayer(nn.Module):
    def __init__(
        self, dim, inner_dim, num_heads, causal=False, attention_method="hyper"
    ):
        super().__init__()
        num_heads = dim // 128
        if attention_method == "hyper":
            self.attn = LLMAttention(dim, dim, num_heads, causal=causal)
        else:
            self.attn = ViTAttention(dim, dim, num_heads, causal=causal)
        self.input_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.post_attention_layernorm = LlamaRMSNorm(dim, eps=1e-05)
        self.mlp = LlamaMLP(dim, inner_dim)
        self.causal = causal

    def forward(self, hidden_states, residual_in=-1):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        if residual_in != -1:
            return hidden_states, 0.0
        return hidden_states


import torch.utils.checkpoint as checkpoint


class LLMClassificationDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes,
        mlp_ratio=4,
        use_checkpoint=False,
        hidden_size=768,
        num_heads=8,
        n_layers=2,
        attention_method=None,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.input_proj = nn.Linear(in_dim, hidden_size)
        assert attention_method in ["hyper", "naive", "mamba"]
        if attention_method == "mamba":
            from ..context_encoders.mamba import create_block

            ssm_cfg = {"d_state": 16}
            self.layers = nn.Sequential(
                *[
                    create_block(
                        d_model=hidden_size,
                        ssm_cfg=ssm_cfg,
                        fused_add_norm=False,
                        residual_in_fp32=True,
                        drop_rate=0.0,
                        drop_path_rate=0.0,
                        reverse=i % 2 == 0,
                        transpose=False,
                        use_mlp=False,
                        is_2d=False,
                        rms_norm=False,
                        split_head=False,
                        use_nd=False,
                        downsample=False,
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            self.layers = nn.Sequential(
                *[
                    LLMLayer(
                        hidden_size,
                        hidden_size * mlp_ratio,
                        num_heads,
                        causal=True,
                        attention_method=attention_method,
                    )
                    for _ in range(n_layers)
                ]
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.hidden_size = hidden_size
        self.cls = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, num_classes)
        )
        self.init_weights()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # if self.pos_embed is not None:
        #     pos_embed = get_2d_sincos_pos_embed(
        #         self.pos_embed.shape[-1],
        #         int(self.patch_embed.num_patches**0.5),
        #         cls_token=True,
        #     )
        #     self.pos_embed.data.copy_(
        #         torch.from_numpy(pos_embed).float().unsqueeze(0)
        #     )
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)

    def forward(self, x, mem):
        x = x[-1]
        n, _, h, w = x.shape
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, h, cls_token=False)
        x = rearrange(x, "n c h w -> n (h w )c")

        # breakpoint()
        x = self.input_proj(x)
        x = x + torch.tensor(pos_embed).to(x)
        x = torch.cat([x, self.cls_token.repeat(n, 1, 1)], dim=1)
        residual = None
        if self.use_checkpoint:
            for i, blk in enumerate(self.layers):
                x, residual = checkpoint.checkpoint(blk, x, residual)
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
        else:
            for i, blk in enumerate(self.layers):
                x, residual = blk(x, residual)
                if i == len(self.layers) - 1:
                    x = (x + residual) if residual is not None else x
        x = self.cls(x[:, -1])  # N C
        return {"label": x}


class EncoderDecoder(AbstractModel):
    def __init__(
        self,
        backbone: nn.Module,
        xl_config: ContextEncoderConfig,
        channels_last: bool = False,
        crop_size: int = 256,
        skip_decoder: bool = False,
        backbone_name: str = "swinv2_tiny_window16_256_timm",
        dataset: str = "inaturalist",
        num_classes: int = 9999,
        mlp_ratio: int = 4,
        cls_head: str = None,
        **kwargs,
    ):
        self.channels_last = channels_last
        self.crop_size = crop_size
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.skip_decoder = skip_decoder
        self.dataset = dataset
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.cls_head = cls_head
        self.grad_ratio = xl_config.grad_ratio
        self.xl_config = xl_config

        super().__init__()

        self.init_decoder()

        self.name = "u-{}".format(backbone_name)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x, mem=tuple(), cord=(0, 0), **kwargs):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x
        n_chip = x.shape[2] // self.crop_size
        # print(n_chip)
        if do_chip := n_chip > 1:  # on gradient chipping
            x = rearrange(
                x,
                "N C (HP HC) (WP WC)-> (N HP WP) C HC WC ",
                HP=n_chip,
                WP=n_chip,
                HC=self.crop_size,
                WC=self.crop_size,
            )
            if self.grad_ratio >= 1.0:
                enc_results = self.encoder(x)
            else:
                n = x.shape[0]
                n_grad = math.ceil(n * self.grad_ratio)
                idx = torch.randperm(n)
                idx_inv = torch.argsort(idx)
                out_grad = self.encoder(x[:n_grad])
                with torch.no_grad():
                    out_stopgrad = self.encoder(x[n_grad:])
                enc_results = list(
                    [
                        torch.cat([a, b], dim=0)[idx_inv]
                        for a, b in zip(out_grad, out_stopgrad)
                    ]
                )
                del out_grad
                del out_stopgrad
            enc_results = list(
                [
                    rearrange(
                        i,
                        "(N HP WP) C HC WC -> N C (HP HC) (WP WC)",
                        HP=n_chip,
                        WP=n_chip,
                    )
                    for i in enc_results
                ]
            )
        else:
            enc_results = self.encoder(x)

        output = self.decoder(enc_results, x_skip)
        return output

    def init_decoder(self):
        if self.dataset == "inaturalist":
            assert self.cls_head in ["naive", "xl"]

            clasifier = (
                ClassificationDecoder
                if self.cls_head == "naive"
                else LLMClassificationDecoder
            )
            extra_kwargs = {}
            if self.cls_head == "xl":
                extra_kwargs = dict(
                    hidden_size=self.xl_config.hidden_size,
                    n_layers=self.xl_config.n_layer,
                    attention_method=self.xl_config.attention_method,
                )
            self.decoder = clasifier(
                in_dim=self.filters[-1],
                num_classes=self.num_classes,
                mlp_ratio=self.mlp_ratio,
                **extra_kwargs,
            )
        else:
            raise Exception("Unknown dataset {}".format(self.dataset))


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio) -> None:
        super().__init__()
        self.hidden_channels = int(mlp_ratio * in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            nn.Linear(in_channels, self.hidden_channels),
            nn.BatchNorm1d(self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class UnetDecoderLastConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, 1),
        )

    def forward(self, x, img=None):
        return self.layer(x)


class UnetDecoderLastConvSkip(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels + 2, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, 1),
        )

    def forward(self, x, img):
        x = self.layer1(x)
        x = torch.cat([x, img], dim=1)
        x = self.layer2(x)
        return x


class MAEPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_classes):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, stride, stride),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, 1),
        )

    def forward(self, x):
        return self.layer(x)
