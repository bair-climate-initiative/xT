from dataclasses import dataclass, field

from .backbones import *
from .transformer_xl import TransformerXLConfig
from .unet import TimmUnet

# from hydra.core.config_store import ConfigStore
# from hydra.utils import instantiate


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
    name: str = "TimmUnet"
    """Name of overarching model architecture."""
    resume: str = ""
    """Path to checkpoint to resume training from. Empty for none."""
    tiling: str = "naive"
    """Transformer-XL tiling strategy"""
    backbone_class: str = "revswinv2_tiny_window16_256_xview"
    """Class name for backbone."""
    patch_size: int = 16
    """Patch sized used for transformer XL.""" # TODO: properly derive this

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    xl_context: TransformerXLConfig = field(default_factory=TransformerXLConfig)


# cs = ConfigStore.instance()
# cs.store(name="config", group="model", node=ModelConfig)


def build_model(config: ModelConfig):
    backbone_class = config.backbone_class
    backbone = eval(backbone_class)(**config.backbone)

    if config.name == "TimmUnet":
        model = TimmUnet(
            backbone=backbone,
            xl_config=config.xl_context,
            channels_last=config.backbone.channel_last,
            crop_size=config.backbone.img_size,
            context_mode=config.xl_context.enabled,
            skip_decoder=False,
            backbone_name=config.backbone_class,
        )
    return model