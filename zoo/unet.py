import os

import timm
import torch.hub
from torch.nn import Dropout2d
from torch.utils import model_zoo

from .revswin import REVSWIN_CFG
from .revswinv2 import REVSWINV2_CFG
from .swin import SWIN_CFG

SWIN_CFG = {**SWIN_CFG, **REVSWIN_CFG, **REVSWINV2_CFG}

from .transformer_xl import MemTransformerLM
from .vit import MAEDecoder
from .vit import registry as VIT_CFG

encoder_params = {
    "resnet34": {"decoder_filters": [48, 96, 176, 192], "last_upsample": 32}
}

default_decoder_filters = [48, 96, 176, 256]
default_last = 48

import torch
import torch.nn.functional as F
import numpy as np

from .lib.hier.utils.patch_embed import PatchEmbed3D,PatchEmbed4D,PatchRecover3D,PatchRecover4D,build_upsample,build_downsample,ConvBlock4D
from .lib.hier.utils.data_utils import parse_lookbacks 
from .lib.hier.utils.blocks import registry as BLOCKS
from torch import nn


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
            in_channels, out_channels, kernel_size=3, dilation=dilation, activation=None
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

    def initialize_encoder(self, model, model_url, num_channels_changed=False):
        if os.path.isfile(model_url):
            pretrained_dict = torch.load(model_url)
        else:
            pretrained_dict = model_zoo.load_url(model_url)
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
            pretrained_dict = {
                k.replace("module.", ""): v for k, v in pretrained_dict.items()
            }
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_names[0] + ".weight"][
                :, :3, ...
            ] = pretrained_dict[self.first_layer_params_names[0] + ".weight"].data
            skip_layers = self.first_layer_params_names
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if not any(k.startswith(s) for s in skip_layers)
            }
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_names(self):
        return ["conv1.conv"]


class TimmUnet(AbstractModel):
    def __init__(
        self,
        encoder="resnet34",
        in_chans=2,
        pretrained=True,
        channels_last=False,
        crop_size = 256,
        context_mode='None',
        skip_decoder=False,
        **kwargs
    ):
        if not hasattr(self, "first_layer_stride_two"):
            self.first_layer_stride_two = True
        if not hasattr(self, "decoder_block"):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, "bottleneck_type"):
            self.bottleneck_type = ConvBottleneck

        backbone_arch = encoder
        self.channels_last = channels_last
        if "swin" in backbone_arch:
            backbone = SWIN_CFG[backbone_arch](
                img_size=crop_size, pretrained=pretrained, input_dim=in_chans, **kwargs
            )
        elif "vit" in backbone_arch:
            backbone = VIT_CFG[backbone_arch](
                img_size=crop_size, pretrained=pretrained, input_dim=in_chans, **kwargs
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
        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last
        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get(
                "decoder_filters", self.filters[:-1]
            )
            self.last_upsample_filters = encoder_params[encoder].get(
                "last_upsample", self.decoder_filters[0] // 2
            )

        super().__init__()
        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )
        self.context_mode = context_mode
        self.extra_context = False
        if context_mode == "transformer_xl":
            self.extra_context = True
            # TODO: build transformer layers
            transformer_xl_config = kwargs["transformer_xl"]
            d_model = self.filters[-1]
            n_length = (self.crop_size // backbone.feature_info[-1]["reduction"]) ** 2
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

        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x,mem=tuple(),**kwargs):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x
        enc_results = self.encoder(x)
        if self.context_mode == "transformer_xl":
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

        fishing_mask = self.fishing_mask(x,x_skip).contiguous(
            memory_format=torch.contiguous_format
        )
        vessel_mask = self.vessel_mask(x,x_skip).contiguous(
            memory_format=torch.contiguous_format
        )
        center_mask = self.center_mask(x,x_skip).contiguous(
            memory_format=torch.contiguous_format
        )
        length_mask = self.length_mask(x,x_skip).contiguous(
            memory_format=torch.contiguous_format
        )
        output = {
            "fishing_mask": fishing_mask,
            "vessel_mask": vessel_mask,
            "center_mask": center_mask,
            "length_mask": length_mask,
        }
        if self.context_mode == "transformer_xl":
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
                img_size=crop_size, pretrained=pretrained, input_dim=in_chans, **kwargs
            )
        elif "vit" in backbone_arch:
            backbone = VIT_CFG[backbone_arch](
                img_size=crop_size, pretrained=pretrained, input_dim=in_chans, **kwargs
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
        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get(
                "decoder_filters", self.filters[:-1]
            )
            self.last_upsample_filters = encoder_params[encoder].get(
                "last_upsample", self.decoder_filters[0] // 2
            )
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
            decoder_embed_dim, self.last_upsample_filters, self.strides[out_indices], 1
        )
        self.center_mask = predictor_cls(
            decoder_embed_dim, self.last_upsample_filters, self.strides[out_indices], 1
        )
        self.length_mask = predictor_cls(
            decoder_embed_dim, self.last_upsample_filters, self.strides[out_indices], 1
        )

        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone
        if context_mode == 'transformer_xl':
            self.extra_context = True
    def forward(self, x,context=None,**kwargs):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.encoder(x,context)
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
            return output,None
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



class HierVitND(AbstractModel):
    def __init__(
        self,
        encoder="resnet34",
        in_chans=2,
        pretrained=True,
        channels_last=False,
        crop_size = 256,
        context_mode='None',
        out_indices=-2,
        decoder_embed_dim=12,
        decoder_depth=8,
        decoder_num_heads=16,
        decoder_mlp_ratio=4.,
        inner_block='ViTBlock',
        blocks='PanguWeatherBlock',
        drop_path=0.1,
        mlp_ratio=4,
        roll=False,
        drop_rate=0.1,
        window_size = (1,1,6,6),
        patch_size = (1,1,16,16),
        inner_bias_mode = 'abs',
        attention_mask = False,
        embed_dim = 768,
        norm_type='LayerNorm',
        block_extr_args=None,
        block_extr_args_swin=None,
        use_checkpoint = False,
        decoder_conv_layers = 2,
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
        #block_extr_args_swin=  []
        # inner_block = None
        # drop_path = None
        # mlp_ratio = None
        #roll = False
        #drop_rate = None
        #window_size = None
        #inner_bias_mode = None
        #attention_mask = None
        # embed_dim = None
        # norm_type = None
        #block_extr_args = None
        # blocks = None
        input_size = (1,1,crop_size,crop_size)
        #patch_size = (1,1,16,16)

        self.input_size = np.array(input_size)
        self.patch_size_4d = np.array(patch_size)
        self.pad_size = (np.ceil(
                (self.patch_size_4d -(self.input_size % self.patch_size_4d )) % self.patch_size_4d 
                / 2
            )).astype(int)
        self.n_patches = ((self.input_size + 2 * self.pad_size) // self.patch_size_4d).astype(int)
        self.input_size = self.input_size.astype(int)

        patch_out_channels = block_extr_args_swin['embed_dim'][0]
        patch_recover_channels = block_extr_args_swin['embed_dim'][0] * 2

        if True: ## swin
            hier_depths = block_extr_args_swin['hier_depths']
            self.total_hierarchy = len(hier_depths)
            down_sample_size = block_extr_args_swin.get('down_sample_size',[2,]*len(hier_depths))
            encoder_layers = []
            decoder_layers = []
            downsample_layers = []
            upsample_layers = []
            reduction_layers = []
            current_patch_size = self.n_patches
            dpr = [x.item() for x in torch.linspace(0, drop_path, sum(block_extr_args_swin['hier_depths']))]
            dpr_decoder = [x.item() for x in torch.linspace(0, drop_path, sum(block_extr_args_swin['decode_depths']))]

            inner_blocks_swin = block_extr_args_swin.get('inner_block',[inner_block] * self.total_hierarchy)
            norm_mode_swin = [dict(norm_mode=x) for x in block_extr_args_swin['norm_mode']]
            block_cls = BLOCKS[blocks]
            
            
            for i in range(self.total_hierarchy):
                dpr_start_idx = sum(block_extr_args_swin['hier_depths'][:i])
                dpr_start_idx_d = sum(block_extr_args_swin['decode_depths'][:i])
                dpr_layer = dpr[dpr_start_idx:dpr_start_idx+block_extr_args_swin['hier_depths'][i]]
                dpr_layer_d = dpr_decoder[dpr_start_idx_d:dpr_start_idx_d+block_extr_args_swin['decode_depths'][i]]
                #breakpoint()
                encoder_layer = block_cls(block_extr_args_swin['embed_dim'][i],
                        block_extr_args_swin['num_heads'][i],
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
                        proj_drop=0.,depth=block_extr_args_swin['hier_depths'][i],use_checkpoint=self.use_checkpoint,**block_extr_args,**norm_mode_swin[i])
                decoder_layer = block_cls(block_extr_args_swin['embed_dim'][i],
                        block_extr_args_swin['num_heads'][i],
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
                        proj_drop=0.,depth=block_extr_args_swin['decode_depths'][i],use_checkpoint=self.use_checkpoint,**block_extr_args,**norm_mode_swin[i])
                encoder_layers.append(encoder_layer)
                decoder_layers.append(decoder_layer)

                if i < self.total_hierarchy -1 : # not last later
                    downsample_layer,down_info = build_downsample(block_extr_args_swin['embed_dim'][i],block_extr_args_swin['embed_dim'][i+1],tuple(current_patch_size),window_size=down_sample_size[i])
                    upsample_layer = build_upsample(block_extr_args_swin['embed_dim'][i],block_extr_args_swin['embed_dim'][i+1],tuple(current_patch_size),window_size=down_sample_size[i])
                    current_patch_size = np.ceil(current_patch_size / down_info['window_size']).astype(int)
                    downsample_layers.append(downsample_layer)
                    upsample_layers.append(upsample_layer)
                    reduction_layers.append(ConvBlock4D(block_extr_args_swin['embed_dim'][i]*2,block_extr_args_swin['embed_dim'][i],depth=block_extr_args_swin['conv_depth_cat'],skip=block_extr_args_swin['skip']))
            self.encoder_layers = nn.ModuleList(encoder_layers)
            self.decoder_layers = nn.ModuleList(encoder_layers)
            self.downsample_layers = nn.ModuleList(downsample_layers)
            self.upsample_layers = nn.ModuleList(upsample_layers)
            self.reduction_layers = nn.ModuleList(reduction_layers)
            self.roll = roll
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

        self.patch_embed_level = PatchEmbed4D(patch_size=patch_size,
                                              padding = self.pad_size,
                                              in_channels=in_chans,
                                              out_channels=patch_out_channels)
        
        self.unpatchify_level = PatchRecover4D(patch_size=patch_size,
                                            padding=self.pad_size,
                                            input_shape=self.input_size,
                                            in_channels=decoder_embed_dim,
                                            out_channels=patch_recover_channels,
                                            decoder_depth=2)
        predictor_cls = MAEPredictor4D
        self.vessel_mask = predictor_cls(
            decoder_embed_dim, decoder_embed_dim,1,1,depth=decoder_conv_layers
        )
        self.fishing_mask = predictor_cls(
            decoder_embed_dim, decoder_embed_dim,1,1,depth=decoder_conv_layers
        )
        self.center_mask = predictor_cls(
            decoder_embed_dim, decoder_embed_dim,1,1,depth=decoder_conv_layers
        )
        self.length_mask = predictor_cls(
            decoder_embed_dim, decoder_embed_dim,1,1,depth=decoder_conv_layers
        )


    def transformer(self,x):
        encs = []
        for i in range(self.total_hierarchy):
            #print(i,x.max(),'before')
            x = self.encoder_layers[i](x)
            #print(i,x.max(),'after')
            encs.append(x)
            if i < self.total_hierarchy - 1:
                x = self.downsample_layers[i](x)
        #decs = []
        for i in reversed(range(self.total_hierarchy)):
            if i < self.total_hierarchy - 1:
                x = self.upsample_layers[i](x)
                x = torch.cat([x,encs[i]],dim=1)
                x = self.reduction_layers[i](x)
            x = self.decoder_layers[i](x)
            #decs.append(x)
        x = torch.cat([x,encs[0]],dim=1) # N C T L H W 
        return x

    def forward(self, x,context=None,**kwargs):
        # Encoder
        ''' N C H W'''
        if len(x.shape) == 4:
            x = x[:,:,None,None] # N C 1 1 H W
        x = self.patch_embed_level(x)  # N C T L H W
        ## transformer

        x = self.transformer(x)

        x = self.unpatchify_level(x)
        #del encs
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



class TimmUnetPANDA(AbstractModel):
    def __init__(
        self,
        encoder="resnet34",
        in_chans=3,
        pretrained=True,
        channels_last=False,
        **kwargs
    ):
        if not hasattr(self, "first_layer_stride_two"):
            self.first_layer_stride_two = True
        if not hasattr(self, "decoder_block"):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, "bottleneck_type"):
            self.bottleneck_type = ConvBottleneck

        backbone_arch = encoder
        self.channels_last = channels_last
        backbone = timm.create_model(
            backbone_arch,
            features_only=True,
            in_chans=in_chans,
            pretrained=pretrained,
            **kwargs
        )
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last
        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get(
                "decoder_filters", self.filters[:-1]
            )
            self.last_upsample_filters = encoder_params[encoder].get(
                "last_upsample", self.decoder_filters[0] // 2
            )

        super().__init__()
        self.bottlenecks = nn.ModuleList(
            [
                self.bottleneck_type(self.filters[-i - 2] + f, f)
                for i, f in enumerate(reversed(self.decoder_filters[:]))
            ]
        )

        self.decoder_stages = nn.ModuleList(
            [self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))]
        )
        self.out_mask = UnetDecoderLastConv(
            self.decoder_filters[0], self.last_upsample_filters, 6
        )
        self.cls_head = MLP(self.filters[-1], 10, 2.0)
        # self.fishing_mask = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, 1)
        # self.center_mask = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, 1)
        # self.length_mask = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, 1)

        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)  # N X C X H X W
        enc_results = self.encoder(x)
        x = enc_results[-1]  # N X C X H X W

        gap_x = x.mean((2, 3))  # N X C
        out_cls = self.cls_head(gap_x)
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = -(idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        out_mask = self.out_mask(x).contiguous(memory_format=torch.contiguous_format)

        return {
            "mask": out_mask,
            "local_score": out_cls,
        }

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
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.SiLU(inplace=True)
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

    def forward(self, x,img=None):
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
            nn.Conv2d(out_channels+2, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, 1),
        )

    def forward(self, x,img):
        x = self.layer1(x)
        x = torch.cat([x,img],dim=1)
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

from .lib.hier.utils.patch_embed import ConvBlock4D,Conv4d



class MAEPredictor4D(nn.Module):
    def __init__(self, in_channels, out_channels, stride,num_classes,depth=2):
        super().__init__()
        self.layer = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, 3,padding=1),
            # nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, 1),
            #ConvBlock4D(in_channels,in_channels,depth,bias=True,skip=True,prediction_channel=num_classes)   
        )

    def forward(self, x):
        x = x[:,:,0,0]
        return self.layer(x).contiguous(
            memory_format=torch.contiguous_format
        ) # NO T L HERE
