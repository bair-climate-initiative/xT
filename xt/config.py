import os
from dataclasses import dataclass, field

from omegaconf import OmegaConf

from .datasets import DataConfig
from .losses import LossConfig
from .models import ModelConfig
from .optimizer import OptimizerConfig

# Get rid of all conflicts


@dataclass
class TrainConfig:
    """Full training config."""

    epochs: int = 120
    """Number of epochs to train for."""
    batch_size: int = 4
    """Batch size per GPU."""
    val_batch_size: int = 2
    """Validation batch size per GPU."""
    freeze_epochs: int = 0
    """Number of epochs to freeze encoder for."""
    freeze_bn: bool = False
    """Whether to freeze batch norm layers."""
    test_every: int = 1
    """Run test every n epochs."""
    test_reset: bool = True
    """Removes existing test csv before testing."""
    clip_grad: float = 5.0
    """Clip gradient norm."""


@dataclass
class MainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    losses: LossConfig = field(default_factory=LossConfig)

    base_configs: list = field(default_factory=list)
    """List of configs to inherit from, in order of override priority."""
    distributed: bool = True
    """Whether to use distributed training."""
    output_dir: str = "outputs/"
    """Output directory for weights, etc.."""
    fp16: bool = False
    """Whether to use mixed precision training."""
    fsdp: bool = False
    """Whether to use Fully Sharded Data Parallel training."""
    test: bool = False
    """Testing only flag."""
    val: bool = False
    """Validation only flag."""
    name: str = ""
    """Run name."""
    eval_sampler: bool = False
    """Use evaluation sampler for validation, i.e. no repeated samples."""


def _merge_configs(cfg: MainConfig, cfg_file: str):
    """Merge config at cfg_file with cfg."""
    other_cfg = OmegaConf.load(cfg_file)

    if hasattr(other_cfg, "base_configs"):  # equiv to .get("base_config", [])
        for base_cfg in other_cfg.base_configs:
            cfg = _merge_configs(cfg, base_cfg)

    if os.environ.get("RANK", "0") == "0":
        # needed since distrbuted not initialized
        print(f"==> Merging config file {cfg_file} into config.")
    cfg = OmegaConf.merge(cfg, other_cfg)
    return cfg


def create_config(schema: MainConfig, cfg_path: str):
    """Create config from input config, recursively resolving base configs."""
    # First resolve input config and it's base_configs
    cfg = _merge_configs(schema, cfg_path)

    return cfg
