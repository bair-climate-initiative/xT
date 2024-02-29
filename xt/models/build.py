from dataclasses import dataclass, field

from .backbones import *
from .context_encoders.transformer_xl import ContextEncoderConfig
from .decoders.unet import EncoderDecoder


@dataclass
class BackboneConfig:
    """Configuration for feature extracting backbone."""

    # _target_: str = "models.backbones.revswinv2_tiny_window16_256_xview"
    # """Fully qualified class name for the backbone to instantiate."""
    # name: str = "revswinv2_tiny"
    # """Shorthand for backbone name."""
    in_chans: int = 3
    """Number of channels in input data."""
    input_dim: int = 2
    """Input dimension."""
    drop_path_rate: float = 0.0
    """Drop path rate for stochastic depth."""
    pretrained: str = ""
    """Path to pretrained weights, empty for none."""
    channel_last: bool = True
    """If channels are last in data format."""
    img_size: int = 256
    """Expected input size of data."""


@dataclass
class ModelConfig:
    name: str = "EncoderDecoder"
    """Name of overarching model architecture."""
    resume: str = ""
    """Path to checkpoint to resume training from. Empty for none."""
    tiling: str = "naive"
    """Transformer-XL tiling strategy"""
    backbone_class: str = "swinv2_tiny_window16_256_timm"
    """Class name for backbone."""
    patch_size: int = 16
    """Patch sized used for transformer XL."""  # TODO: properly derive this
    num_classes: int = 9999
    cls_head: str = "naive"
    """Number of classes for head on dataset."""
    mlp_ratio: int = 4
    """MLP ratio for Enc/Dec."""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    context: ContextEncoderConfig = field(default_factory=ContextEncoderConfig)


def build_model(config: ModelConfig, dataset: str = "inaturalist"):
    backbone_class = config.backbone_class
    backbone = eval(backbone_class)(**config.backbone)

    if config.name == "EncoderDecoder":
        model = EncoderDecoder(
            backbone=backbone,
            xl_config=config.context,
            channels_last=config.backbone.channel_last,
            crop_size=config.backbone.img_size,
            skip_decoder=False,
            backbone_name=config.backbone_class,
            dataset=dataset,
            num_classes=config.num_classes,
            mlp_ratio=config.mlp_ratio,
            cls_head=config.cls_head,
        )
    return model
