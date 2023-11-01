import numpy as np
from einops import rearrange
from torch import nn

from .conv_4d import Conv4d
from .norm_4d import LayerNorm4d


class PatchEmbedND(nn.Module):
    def __init__(
        self, patch_size, in_channels, out_channels, padding=None, bias=True
    ):
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channnnels = out_channels
        self.bias = bias
        self.padding = padding
        super().__init__()
        self.layers = self.build_layers()

    def build_layers(self):
        raise NotImplementedError

    def forward(self, x):
        return self.layers(x)


class PatchEmbed3D(PatchEmbedND):
    def build_layers(self):
        return nn.Conv3d(
            self.in_channels,
            self.out_channnnels,
            self.patch_size,
            stride=self.patch_size,
            bias=self.bias,
            padding=self.padding,
        )


class PatchEmbed4D(PatchEmbedND):
    def build_layers(self):
        return Conv4d(
            self.in_channels,
            self.out_channnnels,
            self.patch_size,
            stride=self.patch_size,
            bias=self.bias,
            padding=self.padding,
        )


class ConvBlock4D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=1,
        bias=True,
        skip=False,
        prediction_channel=None,
    ) -> None:
        super().__init__()
        layers = []
        self.in_channels = in_channels
        self.out_channels = out_channels
        in_ch = self.in_channels
        self.skip_type = skip
        self.depth = depth
        self.prediction_channel = (
            prediction_channel if prediction_channel else out_channels
        )
        if self.depth == 0:
            self.skip = nn.Linear(in_channels, self.prediction_channel)
        elif self.depth == -1:
            pass
        else:
            if not skip:
                self.skip = lambda x: 0.0
            elif self.in_channels == self.out_channels:
                self.skip = nn.Identity()
            else:
                self.skip = Conv4d(
                    in_ch,
                    self.out_channels,
                    (1, 1, 1, 1),
                    stride=(1, 1, 1, 1),
                    bias=bias,
                    padding=(0, 0, 0, 0),
                )
            for i in range(depth):
                layers.append(
                    Conv4d(
                        in_ch,
                        self.out_channels,
                        (3, 3, 3, 3),
                        stride=(1, 1, 1, 1),
                        bias=bias,
                        padding=(1, 1, 1, 1),
                    )
                )
                layers.append(LayerNorm4d(out_channels))
                layers.append(nn.GELU())
                in_ch = self.out_channels
            layers.append(
                Conv4d(
                    in_ch,
                    self.prediction_channel,
                    (1, 1, 1, 1),
                    stride=(1, 1, 1, 1),
                    bias=bias,
                    padding=(0, 0, 0, 0),
                )
            )
            self.layers = nn.Sequential(
                *layers,
            )

    def forward(self, x):
        """
        NCTLHW
        """
        if self.depth == 0:
            return self.skip(x.transpose(1, -1)).transpose(
                1, -1
            )  # +self.layers(x)
        elif self.depth == -1:
            x1, x2 = x.chunk(2, dim=1)
            return x1 + x2
        else:
            return self.skip(x) + self.layers(x)


class PatchRecoverND(nn.Module):
    def __init__(
        self,
        patch_size,
        in_channels,
        out_channels,
        input_shape,
        padding=None,
        bias=True,
        decoder_depth=2,
    ):
        """
        in and out channels are w/ resp to PatchEmbed Layer
        """
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channnnels = out_channels
        self.bias = bias
        self.input_shape = input_shape
        self.padding = padding
        super().__init__()

        self.head = nn.ModuleList()
        for _ in range(decoder_depth):
            self.head.append(nn.Linear(out_channels, out_channels))
            self.head.append(nn.GELU())
        self.head.append(
            nn.Linear(out_channels, self.in_channels * np.product(patch_size))
        )

    def preprocess(self, x):
        raise NotImplementedError

    def post_process(self, x):
        raise NotImplementedError

    def forward(self, x):
        x = self.preprocess(x)
        for layer in self.head:
            x = layer(x)
        x = self.post_process(x)
        return x


class PatchRecover3D(PatchRecoverND):
    def preprocess(self, x):
        """
        X: N C T H W
        """
        x = rearrange(x, "N C T H W -> N T H W C")
        return x

    def post_process(self, x):
        """
        X: N C T H W
        """
        pt, ph, pw = self.patch_szize
        axis_len = dict(
            C=self.in_channels,
            pT=pt,
            pH=ph,
            pW=pw,
        )
        x = rearrange(
            x, "N T H W (C pT pH pW )-> N C (T pT) (H pH) (W pW)", **axis_len
        )
        x = x[
            :,
            :,
            self.padding[0] : self.padding[0] + self.input_shape[0],
            self.padding[1] : self.padding[1] + self.input_shape[1],
            self.padding[2] : self.padding[2] + self.input_shape[2],
        ]
        return x


class PatchRecover4D(PatchRecoverND):
    def preprocess(self, x):
        """
        X: N C T H W
        """
        x = rearrange(x, "N C T L H W -> N T L H W C")
        return x

    def post_process(self, x):
        """
        X: N C T H W
        """
        pt, pl, ph, pw = self.patch_size
        axis_len = dict(
            C=self.in_channels,
            pT=pt,
            pL=pl,
            pH=ph,
            pW=pw,
        )
        x = rearrange(
            x,
            "N T L H W (C pT pL pH pW )-> N C (T pT) (L pL) (H pH) (W pW)",
            **axis_len,
        )
        x = x[
            :,
            :,
            self.padding[0] : self.padding[0] + self.input_shape[0],
            self.padding[1] : self.padding[1] + self.input_shape[1],
            self.padding[2] : self.padding[2] + self.input_shape[2],
            self.padding[3] : self.padding[3] + self.input_shape[3],
        ]
        return x


def compute_downsample_pad(input_shape, window_size, one_sided=False):
    output_pad = []
    for x, y in zip(input_shape, window_size):
        if (reminder := x % y) == 0:
            output_pad.append(0)
        elif one_sided:
            output_pad.append(y - reminder)
        else:
            pad_tgt = np.ceil((y - reminder) / 2.0).astype(int)
            output_pad.append(pad_tgt)
    return output_pad


def build_downsample(
    in_channels, out_channels, input_shape, window_size=(2, 2, 2, 2)
):
    if type(window_size) == int:
        window_size = (window_size, window_size, window_size, window_size)
    window_size = tuple([min(x, y) for x, y in zip(input_shape, window_size)])
    padding = compute_downsample_pad(input_shape, window_size)
    return PatchEmbed4D(
        patch_size=window_size,
        in_channels=in_channels,
        out_channels=out_channels,
        padding=padding,
        bias=True,
    ), dict(window_size=np.array(window_size).astype(float), padding=padding)


def build_upsample(
    in_channels, out_channels, input_shape, window_size=(2, 2, 2, 2)
):
    if type(window_size) == int:
        window_size = (window_size, window_size, window_size, window_size)
    window_size = tuple([min(x, y) for x, y in zip(input_shape, window_size)])
    padding = compute_downsample_pad(input_shape, window_size)
    return PatchRecover4D(
        window_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_shape=input_shape,
        padding=padding,
        bias=True,
        decoder_depth=0,
    )


def crop_4d(x, input_shape, padding=(0, 0, 0, 0)):
    """
    x: N C T L H W
    """
    return x[
        :,
        :,
        padding[0] : padding[0] + input_shape[0],
        padding[1] : padding[1] + input_shape[1],
        padding[2] : padding[2] + input_shape[2],
        padding[3] : padding[3] + input_shape[3],
    ]
