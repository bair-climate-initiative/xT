from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
import wandb
from einops import rearrange
from madgrad import MADGRAD
from matplotlib import pyplot as plt
from timm.models import inception_v3
from timm.optim import AdamW
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, MultiStepLR
from torch.optim.rmsprop import RMSprop
from torch.utils.data import Subset
from warmup_scheduler import GradualWarmupScheduler

from .schedulers import ExponentialLRScheduler, LRStepScheduler, PolyLR
from .utils import get_world_size


@dataclass
class OptimizerConfig:
    """Optimizer Configuation: name, eps, beta, momentum."""

    name: str = "adamw"
    """Optimizer shortname: options are [sgd, adam, adamw]"""
    eps: float = 1e-8
    """Optimizer epsilon."""
    beta: Tuple(float) = (0.9, 0.999)
    """Optimizer betas"""
    momentum: float = 0.9
    """SGD Momentum"""


def create_optimizer(
    optimizer_config: OptimizerConfig,
    model: nn.Module,
    num_samples: int,
):
    """Creates optimizer and schedule from configuration

    Parameters
    ----------
    optimizer_config : dict
        Dictionary containing the configuration options for the optimizer.
    model : Model
        The network model.

    Returns
    -------
    optimizer : Optimizer
        The optimizer.
    scheduler : LRScheduler
        The learning rate scheduler.
    """

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "_bn0.weight",
        "_bn1.weight",
        "_bn2.weight",
    ]

    num_gpus = get_world_size()

    def make_params(param_optimizer, lr=None):
        params = [
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": optimizer_config["weight_decay"],
            },
            {
                "params": [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        for p in params:
            if lr is not None:
                p["lr"] = lr
        return params

    if optimizer_config.train.classifier_multiplier != 1:
        classifier_lr = (
            optimizer_config.train.lr
            * optimizer_config.train.classifier_multiplier
        )
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if not v.requires_grad:
                continue
            if k.find("encoder") != -1:
                net_params.append((k, v))
            else:
                classifier_params.append((k, v))
        params = []

        params.extend(make_params(classifier_params, classifier_lr))
        params.extend(make_params(net_params))
        print("param_groups", len(params))
    else:
        param_optimizer = list(model.named_parameters())
        params = make_params(param_optimizer)
        print("param_groups", len(params))
    train_bs = optimizer_config["train_bs"]
    epochs = optimizer_config["schedule"]["epochs"]
    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            params,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"],
            nesterov=optimizer_config["nesterov"],
        )

    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(
            params,
            eps=optimizer_config.get("eps", 1e-8),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = AdamW(
            params,
            eps=optimizer_config.get("eps", 1e-8),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
    elif optimizer_config["type"] == "RmsProp":
        optimizer = RMSprop(
            params,
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
        )
    elif optimizer_config["type"] == "MadGrad":
        optimizer = MADGRAD(params, lr=optimizer_config["learning_rate"])
    else:
        raise KeyError(
            "unrecognized optimizer {}".format(optimizer_config["type"])
        )

    if optimizer_config["schedule"]["type"] == "step":
        scheduler = LRStepScheduler(
            optimizer, **optimizer_config["schedule"]["params"]
        )
    elif optimizer_config["schedule"]["type"] == "cosine":
        tmax = int(epochs * num_samples / (num_gpus * train_bs))
        eta_min = optimizer_config["schedule"]["params"]["eta_min"]
        print(f"Cosine decay with T_max:{tmax} eta_min:{eta_min}")
        scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)
    elif optimizer_config["schedule"]["type"] == "clr":
        scheduler = CyclicLR(
            optimizer, **optimizer_config["schedule"]["params"]
        )
    elif optimizer_config["schedule"]["type"] == "multistep":
        scheduler = MultiStepLR(
            optimizer, **optimizer_config["schedule"]["params"]
        )
    elif optimizer_config["schedule"]["type"] == "exponential":
        scheduler = ExponentialLRScheduler(
            optimizer, **optimizer_config["schedule"]["params"]
        )
    elif optimizer_config["schedule"]["type"] == "poly":
        scheduler = PolyLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config["schedule"]["type"] == "linear":

        def linear_lr(it):
            return (
                it * optimizer_config["schedule"]["params"]["alpha"]
                + optimizer_config["schedule"]["params"]["beta"]
            )

        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    if optimizer_config["schedule"].get("warmup_epoches", 0):
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=int(
                optimizer_config["schedule"]["warmup_epoches"]
                * num_samples
                / (num_gpus * train_bs)
            ),
            after_scheduler=scheduler,
        )

    return optimizer, scheduler
