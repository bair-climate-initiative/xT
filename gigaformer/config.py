import os
from dataclasses import dataclass, field

from omegaconf import DictConfig, OmegaConf
from typing import Any
from .datasets import DataConfig
from .losses import LossConfig
from .models import ModelConfig
from .optimizer import OptimizerConfig

from .utils import ConflictResolver
# Get rid of all conflicts

def merge_dict(src,tgt):
    if not hasattr(src,'items'):
        return tgt
    new_dict = {}
    for k,v in tgt.items():
        new_dict[k] = v
    for k,v in src.items():
        if k not in new_dict:
            new_dict[k] = v
        else:
            new_dict[k] = merge_dict(src[k],tgt[k])
    return new_dict

def create_config(args: Any):
    # Dict like objects
    """Create config from input config, recursively resolving base configs."""
    config = OmegaConf.load(args.config)
    queue = config.get('base_configs',[])
    seen = set()
    while queue: # DFS
        curr_config = OmegaConf.load(queue.pop(0))
        add_queue = curr_config.get('base_configs',[])
        for parent in add_queue:
            if parent not in seen:
                seen.add(parent)
                queue.insert(0,parent)
        config = merge_dict(curr_config,config)
    config = OmegaConf.create(config)
    del args.config

    # print(OmegaConf.to_yaml(cfg))

    return config


# cs = ConfigStore.instance()
# cs.store(name="config", node=XviewConfig)
# cs.store(name="config", group="train", node=TrainConfig)
