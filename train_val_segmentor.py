import logging
import os
import warnings
from dataclasses import dataclass, field

import hydra
import torch
import torch.distributed
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from training.datasets import DataConfig, create_data_datasets
from training.evaluator import XviewEvaluator
from training.losses import LossConfig
from training.optimizer import OptimizerConfig
from training.trainer import PytorchTrainer, TrainConfig
from training.utils import get_rank, get_world_size, is_main_process
from models import ModelConfig

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120

warnings.filterwarnings("ignore")


@dataclass
class XviewConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    losses: LossConfig = field(default_factory=LossConfig)

    distributed: bool = True
    output_dir: str = "weights/"
    log_dir: str = "logs"
    fp16: bool = False
    fsdp: bool = False
    val: bool = False
    name: str = ""


cs = ConfigStore.instance()
cs.store(name="config", node=XviewConfig)


@hydra.main(config="configs", config_name="base_config")
def main(cfg: XviewConfig) -> None:
    if is_main_process():
        os.makedirs(cfg.log_dir, parents=True, exist_ok=True)
        os.makedirs(cfg.output_dir, parents=True, exist_ok=True)


    data_train, data_val = create_data_datasets(cfg)
    seg_evaluator = XviewEvaluator(cfg)
    trainer = PytorchTrainer(
        config=cfg,
        evaluator=seg_evaluator,
        train_data=data_train,
        val_data=data_val,
    )
    if cfg.test:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data_val,
            shuffle=False,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
        test_loader = DataLoader(
            data_val,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        trainer.validate(test_loader)
        return
    if cfg.val:
        trainer.validate()
        return
    trainer.fit()


if __name__ == "__main__":
    main()
