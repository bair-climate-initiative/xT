import logging
import os
import warnings
from pathlib import Path

import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from gigaformer.config import create_config, XviewConfig
from typing import Any
from gigaformer.datasets import build_loader
from gigaformer.evaluator import build_evaluator 
from gigaformer.trainer import PytorchTrainer
from gigaformer.utils import get_rank, get_world_size, is_main_process

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

os.environ["LOGLEVEL"] = "WARNING"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120

warnings.filterwarnings("ignore")


def main(cfg: XviewConfig) -> None:
    if os.environ.get("RANK", "0") == "0":  
        _make_output_directory_structure(cfg)
        print(OmegaConf.to_yaml(cfg))

    train_data, val_data, train_loader, val_loader, mixup_fn = build_loader(cfg.data, cfg.test)
    seg_evaluator = build_evaluator(cfg)
    trainer = PytorchTrainer(
        config=cfg,
        evaluator=seg_evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        mixup_fn=mixup_fn,
    )

    if is_main_process():
        # os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, cfg.name), exist_ok=True)
        print(OmegaConf.to_yaml(cfg))

    if cfg.test:
        sampler = torch.utils.data.distributed.DistributedSampler(
            val_data,
            shuffle=False,
            num_replicas=get_world_size(),
            rank=get_rank(),
        )
        test_loader = DataLoader(
            val_data,
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


def _make_output_directory_structure(cfg):
    print("Making directories...")
    print(f"Making {cfg.output_dir}")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)

if __name__ == "__main__":
    args = OmegaConf.from_cli()  # first grab from cli to determine config
    schema = OmegaConf.structured(XviewConfig)
    config = create_config(schema, args.config)
    main(config)
