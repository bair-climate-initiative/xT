import timm
from torch import nn


class SwinWrapper(nn.Module):
    def __init__(self, model, hidden_size=768):
        super().__init__()
        self.model = model
        # self.model.head = nn.Identity()
        # self.model.norm = nn.Identity()
        self.feature_info = list(model.feature_info)

    def forward(self, x):
        intermediates = self.model(x)
        intermediates = list([x.permute(0, 3, 1, 2) for x in intermediates])
        return intermediates


class SwinXviewWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.model.head = nn.Identity()
        # self.model.norm = nn.Identity()
        self.feature_info = list(model.feature_info)
        embed_dim = model.layers_0.dim
        for i in self.feature_info:
            i["index"] = i["index"] + 1
        self.feature_info = [
            dict(num_chs=embed_dim, reduction=2, module="patch_embed", index=0)
        ] + self.feature_info
        self.upsample = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)

    def forward(self, x):
        intermediates = self.model(x)
        intermediates = [self.model.patch_embed(x)] + intermediates
        intermediates = list([x.permute(0, 3, 1, 2) for x in intermediates])
        intermediates[0] = self.upsample(intermediates[0])
        return intermediates


def swinv2_tiny_window16_256_timm(*args, **kwargs):
    model = timm.create_model(
        "swinv2_tiny_window16_256.ms_in1k", features_only=True, pretrained=True
    )
    return SwinWrapper(model)


def swinv2_small_window16_256_timm(*args, **kwargs):
    model = timm.create_model(
        "swinv2_small_window16_256.ms_in1k", features_only=True, pretrained=True
    )
    return SwinWrapper(model)


def swinv2_base_window16_256_timm(*args, **kwargs):
    model = timm.create_model(
        "swinv2_base_window16_256.ms_in1k", features_only=True, pretrained=True
    )
    return SwinWrapper(model)


def swinv2_large_window16_256_timm(*args, **kwargs):
    model = timm.create_model(
        "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
        features_only=True,
        pretrained=True,
    )
    return SwinWrapper(model)


def swinv2_tiny_window16_256_timm_xview(img_size, *args, **kwargs):
    opts = {"input_size": (2, img_size, img_size)}
    model = timm.create_model(
        "swinv2_tiny_window16_256.ms_in1k",
        features_only=True,
        pretrained=True,
        in_chans=2,
        pretrained_cfg_overlay=opts,
    )
    return SwinXviewWrapper(model)


def swinv2_small_window16_256_timm_xview(img_size, *args, **kwargs):
    opts = {"input_size": (2, img_size, img_size)}
    model = timm.create_model(
        "swinv2_small_window16_256.ms_in1k",
        features_only=True,
        pretrained=True,
        in_chans=2,
        pretrained_cfg_overlay=opts,
    )
    return SwinXviewWrapper(model)


def swinv2_base_window16_256_timm_xview(img_size, *args, **kwargs):
    opts = {"input_size": (2, img_size, img_size)}
    model = timm.create_model(
        "swinv2_base_window16_256.ms_in1k",
        features_only=True,
        pretrained=True,
        in_chans=2,
        pretrained_cfg_overlay=opts,
    )
    return SwinXviewWrapper(model)


def swinv2_large_window16_256_timm_xview(img_size, *args, **kwargs):
    opts = {"input_size": (2, img_size, img_size)}
    model = timm.create_model(
        "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
        features_only=True,
        pretrained=True,
        in_chans=2,
        pretrained_cfg_overlay=opts,
    )
    return SwinXviewWrapper(model)
