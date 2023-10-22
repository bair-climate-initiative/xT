from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from .unet import TimmUnet

@dataclass
class BackboneConfig:
    """Configuration for feature extracting backbone."""
    _target_: str
    """Fully qualified class name for the backbone to instantiate."""
    name: str 
    """Shorthand for backbone name."""
    in_chans: int = 2
    """Number of channels in input data."""
    drop_path_rate: float = 0.0
    """Drop path rate for stochastic depth."""
    pretrained: str = ""
    """Path to pretrained weights, empty for none."""
    channel_last: bool = True
    """If channels are last in data format."""
    input_size: int = 256
    """Expected input size of data."""


@dataclass
class TransformerXLConfig:
    enabled: bool = False
    """If True, use Transformer-XL as context mode.""" 


@dataclass
class ModelConfig:
    name: str = "TimmUnet"
    """Name of overarching model architecture."""
    resume: str = ""
    """Path to checkpoint to resume training from. Empty for none."""
    tiling: str = "naive"
    """Transformer-XL tiling strategy"""

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    xl_context: TransformerXLConfig = field(default_factory=TransformerXLConfig)


cs = ConfigStore.instance()
cs.store(name="config", group="model", node=ModelConfig)

def build_model(config: ModelConfig):
    # Directly calls the appropriate backbone class
    backbone = instantiate(config.backbone) 

    if config.name == "TimmUnet":
        model = TimmUnet(backbone=backbone, 
                         channels_last=config.backbone.channel_last,
                         crop_size=config.backbone.input_size,
                         context_mode=config.xl_context.enabled,
                         skip_decoder=False)
    return model
