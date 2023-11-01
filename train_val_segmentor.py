import logging
import os
import warnings

# import hydra
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from gigaformer.config import XviewConfig, create_config
from gigaformer.datasets import create_data_datasets
from gigaformer.evaluator import XviewEvaluator
from gigaformer.trainer import PytorchTrainer
from gigaformer.utils import get_rank, get_world_size, is_main_process

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

os.environ["LOGLEVEL"] = "WARNING"
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120

warnings.filterwarnings("ignore")


# @hydra.main(config_path="config", config_name="base_config")
def main(cfg: XviewConfig) -> None:
    data_train, data_val = create_data_datasets(cfg.data, cfg.test)
    seg_evaluator = XviewEvaluator(cfg)
    trainer = PytorchTrainer(
        config=cfg,
        evaluator=seg_evaluator,
        train_data=data_train,
        val_data=data_val,
    )

    if is_main_process():
        os.makedirs(cfg.output_dir, exist_ok=True)
        print(OmegaConf.to_yaml(cfg))

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
    args = OmegaConf.from_cli()  # first grab from cli to determine config
    schema = OmegaConf.structured(XviewConfig)
    config = create_config(schema, args)
    main(config)
