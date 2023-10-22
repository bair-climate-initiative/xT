import logging
import os
import warnings

import hydra
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from training.config import XviewConfig
from training.datasets import create_data_datasets
from training.evaluator import XviewEvaluator
from training.trainer import PytorchTrainer
from training.utils import get_rank, get_world_size, is_main_process

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120

warnings.filterwarnings("ignore")


@hydra.main(config_path="config", config_name="base_config")
def main(cfg: XviewConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if is_main_process():
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.output_dir, exist_ok=True)

    data_train, data_val = create_data_datasets(cfg.data)
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
