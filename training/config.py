from dataclasses import dataclass, field

# from hydra.core.config_store import ConfigStore

# from training.trainer import TrainConfig
from models import ModelConfig
from training.datasets import DataConfig
from training.losses import LossConfig
from training.optimizer import OptimizerConfig


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


@dataclass
class XviewConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    losses: LossConfig = field(default_factory=LossConfig)

    distributed: bool = True
    """Whether to use distributed training."""
    output_dir: str = "outputs/"
    """Output directory for weights, etc.."""
    log_dir: str = "logs/"
    """Log directory for training info."""
    fp16: bool = False
    fsdp: bool = False
    """Whether to use Fully Sharded Data Parallel training."""
    test: bool = False
    """Testing only flag."""
    val: bool = False
    """Validation only flag."""
    name: str = ""
    """Run name."""


# cs = ConfigStore.instance()
# cs.store(name="config", node=XviewConfig)
# cs.store(name="config", group="train", node=TrainConfig)
