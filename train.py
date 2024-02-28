import logging
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from xt.config import MainConfig, create_config
from xt.datasets import build_loader
from xt.evaluator import build_evaluator
from xt.trainer import PytorchTrainer
from xt.utils import get_rank, get_world_size, is_main_process

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

os.environ["LOGLEVEL"] = "WARNING"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120

warnings.filterwarnings("ignore")


def main(cfg: MainConfig = None, args=None) -> None:
    if os.environ.get("RANK", "0") == "0":
        _make_output_directory_structure(cfg)

    process_group = _init_distributed(cfg)
    cfg.name = getattr(args, "name", cfg.name)

    train_data, val_data, train_loader, val_loader, mixup_fn = build_loader(
        cfg.data, cfg.test
    )
    seg_evaluator = build_evaluator(cfg)
    trainer = PytorchTrainer(
        config=cfg,
        evaluator=seg_evaluator,
        train_loader=train_loader,
        val_loader=val_loader,
        mixup_fn=mixup_fn,
        process_group=process_group,
    )
    cfg = trainer.config  # Sync configuration with trainer updates

    if is_main_process():
        os.makedirs(os.path.join(cfg.output_dir, cfg.name), exist_ok=True)
        print(OmegaConf.to_yaml(cfg))

    if hasattr(args, "summary") and args.summary:
        return
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


def _init_distributed(config):
    if config.distributed:
        pg = dist.init_process_group(
            backend="nccl",
            # rank=self.config.local_rank, set to torchrun
            # world_size=self.config.world_size,
        )
        if get_rank() == 0:
            print(f"Setting rank. Rank is {get_rank()}")
            print(
                f"There are {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]}"
            )
        torch.cuda.set_device(get_rank())
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu

    return pg


def _make_output_directory_structure(cfg):
    print("Making directories...")
    print(f"Making {cfg.output_dir}")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)


if __name__ == "__main__":
    args = OmegaConf.from_cli()  # first grab from cli to determine config
    schema = OmegaConf.structured(MainConfig)
    config = create_config(schema, args.config)
    main(config, args)
