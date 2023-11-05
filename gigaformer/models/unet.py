import timm
import torch.hub
from torch.nn import Dropout2d

from .backbones.vit import MAEDecoder
from .backbones.vit import registry as VIT_CFG
from .transformer_xl import MemTransformerLM, TransformerXLConfig

default_decoder_filters = [48, 96, 176, 256]
default_last = 48

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .lib.hier.utils.blocks import registry as BLOCKS
from .lib.hier.utils.patch_embed import (
    ConvBlock4D,
    PatchEmbed4D,
    PatchRecover4D,
    build_downsample,
    build_upsample,
)


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


class TimmUnet(AbstractModel):
    def __init__(
        self,
        backbone: nn.Module,
        xl_config: TransformerXLConfig,
        channels_last: bool = False,
        crop_size: int = 256,
        skip_decoder: bool = False,
        backbone_name: str = "revswinv2_tiny",
        **kwargs
    ):
        # if not hasattr(self, "first_layer_stride_two"):
        self.first_layer_stride_two = True
        # if not hasattr(self, "decoder_block"):
        self.decoder_block = UnetDecoderBlock
        # if not hasattr(self, "bottleneck_type"):
        self.bottleneck_type = ConvBottleneck
        self.channels_last = channels_last
        self.crop_size = crop_size
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last

        super().__init__()
        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )
        self.context_mode = xl_config.enabled
        if self.context_mode:
            # TODO: build transformer layers
            transformer_xl_config = {
                "no_memory": xl_config.no_memory,
                "n_layer": xl_config.n_layer,
            }

            d_model = self.filters[-1]
            n_length = (
                self.crop_size // backbone.feature_info[-1]["reduction"]
            ) ** 2
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
                mem_len=n_length,
                cutoffs=cutoffs,
                attn_type=0,
            )

            xl_args.update(transformer_xl_config)
            self.transformer_xl_layers = MemTransformerLM(**xl_args)

        self.decoder_stages = nn.ModuleList(
            [
                self.get_decoder(idx)
                for idx in range(0, len(self.decoder_filters))
            ]
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

        self.name = "u-{}".format(backbone_name)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x, mem=tuple(), **kwargs):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x
        enc_results = self.encoder(x)
        if self.context_mode:
            xx = enc_results[-1]  # N C H W
            old_shape = xx.shape
            xx = xx.flatten(2).permute(2, 0, 1)  # L N C
            xx = self.transformer_xl_layers(xx, xx, *mem)
            pred_out, mem = xx[0], xx[1:]
            pred_out = pred_out.permute(1, 2, 0).view(*old_shape)
            enc_results[-1] = pred_out  # overwrite
            mem = mem
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
        if self.context_mode:
            return output, mem
        else:
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


class EncoderDecoder(AbstractModel):
    def __init__(
        self,
        encoder="resnet34",
        in_chans=2,
        pretrained=True,
        channels_last=False,
        crop_size=256,
        context_mode="None",
        out_indices=-2,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_mlp_ratio=4.0,
        **kwargs
    ):
        backbone_arch = encoder
        self.channels_last = channels_last
        if "swin" in backbone_arch:
            backbone = SWIN_CFG[backbone_arch](
                img_size=crop_size,
                pretrained=pretrained,
                input_dim=in_chans,
                **kwargs
            )
        elif "vit" in backbone_arch:
            backbone = VIT_CFG[backbone_arch](
                img_size=crop_size,
                pretrained=pretrained,
                input_dim=in_chans,
                **kwargs
            )
        else:
            backbone = timm.create_model(
                backbone_arch,
                features_only=True,
                in_chans=in_chans,
                pretrained=pretrained,
                **kwargs
            )
        self.crop_size = crop_size
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.strides = [f["reduction"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last

        if kwargs.get("expected_stride"):
            assert kwargs.get("expected_stride") == self.strides[out_indices]

        super().__init__()
        self.decoder_layers = MAEDecoder(
            embed_dim=self.filters[out_indices],
            decoder_embed_dim=decoder_embed_dim,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=decoder_mlp_ratio,
            num_patches=int((crop_size // self.strides[out_indices]) ** 2),
            decoder_depth=8,
        )
        self.out_indices = out_indices
        self.context_mode = context_mode
        self.extra_context = False
        predictor_cls = MAEPredictor
        self.vessel_mask = predictor_cls(
            decoder_embed_dim,
            self.last_upsample_filters,
            self.strides[out_indices],
            1,
        )
        self.fishing_mask = predictor_cls(
            decoder_embed_dim,
            self.last_upsample_filters,
            self.strides[out_indices],
            1,
        )
        self.center_mask = predictor_cls(
            decoder_embed_dim,
            self.last_upsample_filters,
            self.strides[out_indices],
            1,
        )
        self.length_mask = predictor_cls(
            decoder_embed_dim,
            self.last_upsample_filters,
            self.strides[out_indices],
            1,
        )

        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone
        if context_mode == "transformer_xl":
            self.extra_context = True

    def forward(self, x, context=None, **kwargs):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.encoder(x, context)
        x = enc_results[self.out_indices]
        x = self.decoder_layers(x)
        fishing_mask = self.fishing_mask(x).contiguous(
            memory_format=torch.contiguous_format
        )
        vessel_mask = self.vessel_mask(x).contiguous(
            memory_format=torch.contiguous_format
        )
        center_mask = self.center_mask(x).contiguous(
            memory_format=torch.contiguous_format
        )
        length_mask = self.length_mask(x).contiguous(
            memory_format=torch.contiguous_format
        )
        output = {
            "fishing_mask": fishing_mask,
            "vessel_mask": vessel_mask,
            "center_mask": center_mask,
            "length_mask": length_mask,
        }
        if context:
            return output, None
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
        **kwargs
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
            [
                self.get_decoder(idx)
                for idx in range(0, len(self.decoder_filters))
            ]
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
        **kwargs
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
            [
                self.get_decoder(idx)
                for idx in range(0, len(self.decoder_filters))
            ]
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

        seg = self.seg(x, x_skip).contiguous(
            memory_format=torch.contiguous_format
        )
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
            mem_len=n_length,
            cutoffs=cutoffs,
            attn_type=0,
        )

        xl_args.update(transformer_xl_config)
        self.model = MemTransformerLM(**xl_args)

    def forward(self, enc_results, mem):
        xx = enc_results[-1]  # N C H W
        old_shape = xx.shape
        xx = xx.flatten(2).permute(2, 0, 1)  # L N C
        xx = self.model(xx, xx, *mem)
        pred_out, mem = xx[0], xx[1:]
        pred_out = pred_out.permute(1, 2, 0).view(*old_shape)
        enc_results[-1] = pred_out  # overwrite
        mem = mem
        return enc_results, mem


class ClassificationDecoder(nn.Module):

    def __init__(self,in_dim,num_classes,mlp_ratio=4):
        super().__init__()
        self.layers =  nn.Sequential(
            nn.Linear(in_dim,in_dim*mlp_ratio),
            nn.GELU(),
            nn.LayerNorm(in_dim*mlp_ratio),
            nn.Linear(in_dim*mlp_ratio,num_classes),
        )

    def forward(self, enc_results, mem):
        xx = enc_results[-1]  # N C H W
        xx = xx.mean((-1,-2)) # GAP->N X C
        logits = self.layers(xx) # CLS
        return {'label':logits}

from typing import Any
class EncoderDecoderV2(AbstractModel):
    def __init__(
        self,
        backbone: nn.Module,
        xl_config: TransformerXLConfig,
        channels_last: bool = False,
        crop_size: int = 256,
        skip_decoder: bool = False,
        backbone_name: str = "revswinv2_tiny",
        dataset: str = 'xview3',
        num_classes: int = 9999,
        mlp_ratio: int = 4,
        **kwargs
    ):
        # if not hasattr(self, "first_layer_stride_two"):
        # if not hasattr(self, "decoder_block"):
        self.channels_last = channels_last
        self.crop_size = crop_size
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.skip_decoder = skip_decoder
        self.dataset = dataset
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio

        super().__init__()

        self.context_mode = xl_config.enabled
        if self.context_mode:
            self.init_context_model(xl_config, backbone)

        self.init_decoder()

        self.name = "u-{}".format(backbone_name)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def init_context_model(self, xl_config, backbone):
        self.context_model = TransformerXLContextModel(
            xl_config,
            self.crop_size,
            backbone.feature_info[-1]["reduction"],
            d_model=self.filters[-1],
        )

    def forward(self, x, mem=tuple(), **kwargs):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x
        enc_results = self.encoder(x)
        if self.context_mode:
            enc_results, mem = self.context_model(enc_results, mem)
        output = self.decoder(enc_results, x_skip)
        if self.context_mode:
            return output, mem
        else:
            return output

    def init_decoder(self):
        if self.task == 'xview':
            self.decoder = UNetDecoder(
                ConvBottleneck,
                self.filters,
                UnetDecoderBlock,
                self.decoder_filters,
                self.skip_decoder,
                default_last,
            )
        elif self.task == 'classification':
            self.decoder = ClassificationDecoder(
                in_dim=self.filters[-1],
                num_classes=self.num_classes,
                mlp_ratio=self.mlp_ratio
            )
        else:
            raise NotImplemented


class HierVitND(AbstractModel):
    def __init__(
        self,
        encoder="resnet34",
        in_chans=2,
        pretrained=True,
        channels_last=False,
        crop_size=256,
        context_mode="None",
        out_indices=-2,
        decoder_embed_dim=12,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_mlp_ratio=4.0,
        inner_block="ViTBlock",
        blocks="PanguWeatherBlock",
        drop_path=0.1,
        mlp_ratio=4,
        roll=False,
        drop_rate=0.1,
        window_size=(1, 1, 6, 6),
        patch_size=(1, 1, 16, 16),
        inner_bias_mode="abs",
        attention_mask=False,
        embed_dim=768,
        norm_type="LayerNorm",
        block_extr_args=None,
        block_extr_args_swin=None,
        use_checkpoint=False,
        decoder_conv_layers=2,
        **kwargs
    ):
        super().__init__()
        self.use_checkpoint = True
        backbone_arch = encoder
        self.channels_last = channels_last
        # if 'swin' in backbone_arch:
        #     backbone = SWIN_CFG[backbone_arch](img_size=crop_size,pretrained=pretrained,
        #                                        input_dim=in_chans,
        #                                        **kwargs)
        # elif 'vit' in backbone_arch:
        #     backbone = VIT_CFG[backbone_arch](img_size=crop_size,
        #                                            pretrained=pretrained,
        #                                         input_dim=in_chans,
        #                                        **kwargs)
        # else:
        #     backbone = timm.create_model(
        #         backbone_arch,
        #         features_only=True,
        #         in_chans=in_chans,
        #         pretrained=pretrained,
        #         **kwargs
        #     )
        # self.filters = [f["num_chs"] for f in backbone.feature_info]
        # self.strides = [f['reduction'] for f in backbone.feature_info]

        self.crop_size = crop_size
        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last
        self.extra_context = False
        # TODO: Args
        # block_extr_args_swin=  []
        # inner_block = None
        # drop_path = None
        # mlp_ratio = None
        # roll = False
        # drop_rate = None
        # window_size = None
        # inner_bias_mode = None
        # attention_mask = None
        # embed_dim = None
        # norm_type = None
        # block_extr_args = None
        # blocks = None
        input_size = (1, 1, crop_size, crop_size)
        # patch_size = (1,1,16,16)

        self.input_size = np.array(input_size)
        self.patch_size_4d = np.array(patch_size)
        self.pad_size = (
            np.ceil(
                (self.patch_size_4d - (self.input_size % self.patch_size_4d))
                % self.patch_size_4d
                / 2
            )
        ).astype(int)
        self.n_patches = (
            (self.input_size + 2 * self.pad_size) // self.patch_size_4d
        ).astype(int)
        self.input_size = self.input_size.astype(int)

        patch_out_channels = block_extr_args_swin["embed_dim"][0]
        patch_recover_channels = block_extr_args_swin["embed_dim"][0] * 2

        if True:  # swin
            hier_depths = block_extr_args_swin["hier_depths"]
            self.total_hierarchy = len(hier_depths)
            down_sample_size = block_extr_args_swin.get(
                "down_sample_size",
                [
                    2,
                ]
                * len(hier_depths),
            )
            encoder_layers = []
            decoder_layers = []
            downsample_layers = []
            upsample_layers = []
            reduction_layers = []
            current_patch_size = self.n_patches
            dpr = [
                x.item()
                for x in torch.linspace(
                    0, drop_path, sum(block_extr_args_swin["hier_depths"])
                )
            ]
            dpr_decoder = [
                x.item()
                for x in torch.linspace(
                    0, drop_path, sum(block_extr_args_swin["decode_depths"])
                )
            ]

            inner_blocks_swin = block_extr_args_swin.get(
                "inner_block", [inner_block] * self.total_hierarchy
            )
            norm_mode_swin = [
                dict(norm_mode=x) for x in block_extr_args_swin["norm_mode"]
            ]
            block_cls = BLOCKS[blocks]

            for i in range(self.total_hierarchy):
                dpr_start_idx = sum(block_extr_args_swin["hier_depths"][:i])
                dpr_start_idx_d = sum(block_extr_args_swin["decode_depths"][:i])
                dpr_layer = dpr[
                    dpr_start_idx : dpr_start_idx
                    + block_extr_args_swin["hier_depths"][i]
                ]
                dpr_layer_d = dpr_decoder[
                    dpr_start_idx_d : dpr_start_idx_d
                    + block_extr_args_swin["decode_depths"][i]
                ]
                encoder_layer = block_cls(
                    block_extr_args_swin["embed_dim"][i],
                    block_extr_args_swin["num_heads"][i],
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr_layer,
                    norm_layer=BLOCKS[norm_type],
                    inner_block=BLOCKS[inner_blocks_swin[i]],
                    bias_mode=inner_bias_mode,
                    mask=attention_mask,
                    drop=drop_rate,
                    window_size=window_size,
                    num_patches=current_patch_size,
                    proj_drop=0.0,
                    depth=block_extr_args_swin["hier_depths"][i],
                    use_checkpoint=self.use_checkpoint,
                    **block_extr_args,
                    **norm_mode_swin[i]
                )
                decoder_layer = block_cls(
                    block_extr_args_swin["embed_dim"][i],
                    block_extr_args_swin["num_heads"][i],
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=dpr_layer_d,
                    norm_layer=BLOCKS[norm_type],
                    inner_block=BLOCKS[inner_blocks_swin[i]],
                    bias_mode=inner_bias_mode,
                    mask=attention_mask,
                    drop=drop_rate,
                    window_size=window_size,
                    num_patches=current_patch_size,
                    proj_drop=0.0,
                    depth=block_extr_args_swin["decode_depths"][i],
                    use_checkpoint=self.use_checkpoint,
                    **block_extr_args,
                    **norm_mode_swin[i]
                )
                encoder_layers.append(encoder_layer)
                decoder_layers.append(decoder_layer)

                if i < self.total_hierarchy - 1:  # not last later
                    downsample_layer, down_info = build_downsample(
                        block_extr_args_swin["embed_dim"][i],
                        block_extr_args_swin["embed_dim"][i + 1],
                        tuple(current_patch_size),
                        window_size=down_sample_size[i],
                    )
                    upsample_layer = build_upsample(
                        block_extr_args_swin["embed_dim"][i],
                        block_extr_args_swin["embed_dim"][i + 1],
                        tuple(current_patch_size),
                        window_size=down_sample_size[i],
                    )
                    current_patch_size = np.ceil(
                        current_patch_size / down_info["window_size"]
                    ).astype(int)
                    downsample_layers.append(downsample_layer)
                    upsample_layers.append(upsample_layer)
                    reduction_layers.append(
                        ConvBlock4D(
                            block_extr_args_swin["embed_dim"][i] * 2,
                            block_extr_args_swin["embed_dim"][i],
                            depth=block_extr_args_swin["conv_depth_cat"],
                            skip=block_extr_args_swin["skip"],
                        )
                    )
            self.encoder_layers = nn.ModuleList(encoder_layers)
            self.decoder_layers = nn.ModuleList(encoder_layers)
            self.downsample_layers = nn.ModuleList(downsample_layers)
            self.upsample_layers = nn.ModuleList(upsample_layers)
            self.reduction_layers = nn.ModuleList(reduction_layers)
            self.roll = roll
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

        self.patch_embed_level = PatchEmbed4D(
            patch_size=patch_size,
            padding=self.pad_size,
            in_channels=in_chans,
            out_channels=patch_out_channels,
        )

        self.unpatchify_level = PatchRecover4D(
            patch_size=patch_size,
            padding=self.pad_size,
            input_shape=self.input_size,
            in_channels=decoder_embed_dim,
            out_channels=patch_recover_channels,
            decoder_depth=2,
        )
        predictor_cls = MAEPredictor4D
        self.vessel_mask = predictor_cls(
            decoder_embed_dim,
            decoder_embed_dim,
            1,
            1,
            depth=decoder_conv_layers,
        )
        self.fishing_mask = predictor_cls(
            decoder_embed_dim,
            decoder_embed_dim,
            1,
            1,
            depth=decoder_conv_layers,
        )
        self.center_mask = predictor_cls(
            decoder_embed_dim,
            decoder_embed_dim,
            1,
            1,
            depth=decoder_conv_layers,
        )
        self.length_mask = predictor_cls(
            decoder_embed_dim,
            decoder_embed_dim,
            1,
            1,
            depth=decoder_conv_layers,
        )

    def transformer(self, x):
        encs = []
        for i in range(self.total_hierarchy):
            # print(i,x.max(),'before')
            x = self.encoder_layers[i](x)
            # print(i,x.max(),'after')
            encs.append(x)
            if i < self.total_hierarchy - 1:
                x = self.downsample_layers[i](x)
        # decs = []
        for i in reversed(range(self.total_hierarchy)):
            if i < self.total_hierarchy - 1:
                x = self.upsample_layers[i](x)
                x = torch.cat([x, encs[i]], dim=1)
                x = self.reduction_layers[i](x)
            x = self.decoder_layers[i](x)
            # decs.append(x)
        x = torch.cat([x, encs[0]], dim=1)  # N C T L H W
        return x

    def forward(self, x, context=None, **kwargs):
        # Encoder
        """N C H W"""
        if len(x.shape) == 4:
            x = x[:, :, None, None]  # N C 1 1 H W
        x = self.patch_embed_level(x)  # N C T L H W
        # transformer

        x = self.transformer(x)

        x = self.unpatchify_level(x)
        # del encs
        fishing_mask = self.fishing_mask(x)
        vessel_mask = self.vessel_mask(x)
        center_mask = self.center_mask(x)
        length_mask = self.length_mask(x)
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


from .lib.hier.utils.patch_embed import ConvBlock4D


class MAEPredictor4D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_classes, depth=2):
        super().__init__()
        self.layer = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 3,padding=1),
            # nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
            # ConvBlock4D(in_channels,in_channels,depth,bias=True,skip=True,prediction_channel=num_classes)
        )

    def forward(self, x):
        x = x[:, :, 0, 0]
        return self.layer(x).contiguous(
            memory_format=torch.contiguous_format
        )  # NO T L HERE
