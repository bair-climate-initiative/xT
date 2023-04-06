import os

import timm
import torch.hub
from torch.nn import Dropout2d
from torch.utils import model_zoo

encoder_params = {
    "resnet34": {
        "decoder_filters": [48, 96, 176, 192],
        "last_upsample": 32
    }
}

default_decoder_filters = [48, 96, 176, 256]
default_last = 48

import torch
from torch import nn
import torch.nn.functional as F


class BasicConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, activation=nn.SiLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                            bias=bias)
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
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=None, bias=bias)


class Conv3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=None)


class ConvSilu1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=nn.SiLU)


class ConvSilu3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=nn.SiLU)


class BasicUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.SiLU, mode='nearest'):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * 1
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1)
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
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if num_channels_changed:
            model.state_dict()[self.first_layer_params_names[0] + '.weight'][:, :3, ...] = pretrained_dict[
                self.first_layer_params_names[0] + '.weight'].data
            skip_layers = self.first_layer_params_names
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               not any(k.startswith(s) for s in skip_layers)}
        model.load_state_dict(pretrained_dict, strict=False)

    @property
    def first_layer_params_names(self):
        return ['conv1.conv']


class TimmUnet(AbstractModel):
    def __init__(self, encoder='resnet34', in_chans=2, pretrained=True, channels_last=False,
                 **kwargs):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        backbone_arch = encoder
        self.channels_last = channels_last
        backbone = timm.create_model(backbone_arch, features_only=True, in_chans=in_chans, pretrained=pretrained,
                                     **kwargs)
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last
        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get('decoder_filters', self.filters[:-1])
            self.last_upsample_filters = encoder_params[encoder].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])
        self.vessel_mask = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, 1)
        self.fishing_mask = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, 1)
        self.center_mask = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, 1)
        self.length_mask = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, 1)

        self.name = "u-{}".format(encoder)

        self._initialize_weights()
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x):
        # Encoder
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        enc_results = self.encoder(x)
        x = enc_results[-1]
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        fishing_mask = self.fishing_mask(x).contiguous(memory_format=torch.contiguous_format)
        vessel_mask = self.vessel_mask(x).contiguous(memory_format=torch.contiguous_format)
        center_mask = self.center_mask(x).contiguous(memory_format=torch.contiguous_format)
        length_mask = self.length_mask(x).contiguous(memory_format=torch.contiguous_format)
        return {
            "fishing_mask": fishing_mask,
            "vessel_mask": vessel_mask,
            "center_mask": center_mask,
            "length_mask": length_mask
        }

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])



class TimmUnetPANDA(AbstractModel):
    def __init__(self, encoder='resnet34', in_chans=3, pretrained=True, channels_last=False,
                 **kwargs):
        if not hasattr(self, 'first_layer_stride_two'):
            self.first_layer_stride_two = True
        if not hasattr(self, 'decoder_block'):
            self.decoder_block = UnetDecoderBlock
        if not hasattr(self, 'bottleneck_type'):
            self.bottleneck_type = ConvBottleneck

        backbone_arch = encoder
        self.channels_last = channels_last
        backbone = timm.create_model(backbone_arch, features_only=True, in_chans=in_chans, pretrained=pretrained,
                                     **kwargs)
        self.filters = [f["num_chs"] for f in backbone.feature_info]
        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last
        if encoder in encoder_params:
            self.decoder_filters = encoder_params[encoder].get('decoder_filters', self.filters[:-1])
            self.last_upsample_filters = encoder_params[encoder].get('last_upsample', self.decoder_filters[0] // 2)

        super().__init__()
        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])
        self.out_mask = UnetDecoderLastConv(self.decoder_filters[0], self.last_upsample_filters, 6)
        self.cls_head = MLP(self.filters[-1],10,2.0)
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
            x = x.contiguous(memory_format=torch.channels_last) # N X C X H X W
        enc_results = self.encoder(x)
        x = enc_results[-1] # N X C X H X W

        gap_x = x.mean((2,3)) # N X C
        out_cls = self.cls_head(gap_x)
        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        out_mask = self.out_mask(x).contiguous(memory_format=torch.contiguous_format)
        
        return {
            "mask": out_mask,
            "local_score":  out_cls,
        }

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])

class MLP(nn.Module):

    def __init__(self, in_channels, out_channels,mlp_ratio) -> None:
        super().__init__()
        self.hidden_channels = int(mlp_ratio*in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            nn.Linear(in_channels,self.hidden_channels),
            nn.BatchNorm1d(self.hidden_channels),
            nn.SiLU(),
            nn.Linear(self.hidden_channels,out_channels)
        )

    def forward(self, x):
        return self.layers(x)

class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
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
            nn.SiLU(inplace=True)
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
            nn.Conv2d(out_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.layer(x)


