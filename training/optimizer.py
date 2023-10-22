from dataclasses import dataclass
from typing import Tuple

from hydra.core.config_store import ConfigStore
from timm.optim import AdamW
from torch import nn, optim
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, MultiStepLR
from warmup_scheduler import GradualWarmupScheduler

from .schedulers import ExponentialLRScheduler, LRStepScheduler, PolyLR
from ..train_val_segmentor import TrainConfig
from .utils import get_world_size


@dataclass
class OptimizerConfig:
    """Optimizer Configuation: name, eps, beta, momentum."""

    name: str = "adamw"
    """Optimizer shortname: options are [sgd, adam, adamw]"""

@dataclass 
class SGDConfig:
    """SGD Configuration: momentum, nesterov."""

    momentum: float = 0.9
    """SGD Momentum"""
    nesterov: bool = True
    """SGD Nesterov momentum"""

@dataclass 
class AdamConfig:
    """Adam Configuration: eps, beta."""

    eps: float = 1e-8
    """Adam epsilon."""
    beta: Tuple(float) = (0.9, 0.999)
    """Adam betas"""

@dataclass 
class AdamWConfig:
    """AdamW Configuration: eps, beta."""

    _target_: str = "optim.AdamW"
    """Class Name for instantiation"""
    eps: float = 1e-8
    """Adam epsilon."""
    beta: Tuple(float) = (0.9, 0.999)
    """Adam betas"""


cs = ConfigStore.instance()
cs.store(group="optimizer", name="optimizer", node=OptimizerConfig)
cs.store(group="optimizer", name="sgd", node=SGDConfig)
cs.store(group="optimizer", name="adam", node=AdamConfig)
cs.store(group="optimizer", name="adamw", node=AdamWConfig)


def create_optimizer(
    config: TrainConfig,
    model: nn.Module,
    num_samples: int,
):
    """Creates optimizer and schedule from configuration

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
                "weight_decay": config.train.weight_decay,
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

    if config.train.classifier_ratio != 1.0:
        classifier_lr = (
            config.train.lr
            * config.train.classifier_ratio
        )
        # Separate classifier parameters from all others
        net_params = []
        classifier_params = []
        for k, v in model.named_parameters():
            if not v.requires_grad:
                continue
            if k.find("backbone") != -1:
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
    train_bs = config.train.batch_size
    epochs = config.train.epochs
    optimizer_name = str.lower(config.optimizer.name)
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            params,
            lr=config.train.lr,
            momentum=config.optimizer.momentum,
            nesterov=config.optimizer.nesterov
        )
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(
            params,
            lr=config.train.lr,
            eps=config.optimizer.eps,
            weight_decay=config.train.weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = AdamW(
            params,
            lr=config.train.lr,
            eps=config.optimizer.eps,
            weight_decay=config.train.weight_decay
        )
    else:
        raise KeyError(
            "unrecognized optimizer {}".format(optimizer_name)
        )

    scheduler_name = config.optimizer.scheduler.name
    eta_min = config.train.lr * config.train.min_lr_ratio
    if scheduler_name == "step":
        scheduler = LRStepScheduler(optimizer, eta_min=eta_min)
    elif scheduler_name == "cosine":
        tmax = int(epochs * num_samples / (num_gpus * train_bs))
        print(f"Cosine decay with T_max:{tmax} eta_min:{eta_min}")
        scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)
    # elif scheduler_name == "clr":
    #     scheduler = CyclicLR(
    #         optimizer, **optimizer_config["schedule"]["params"]
    #     )
    # elif scheduler_name == "multistep":
    #     scheduler = MultiStepLR(
    #         optimizer, **optimizer_config["schedule"]["params"]
    #     )
    # elif scheduler_name == "exponential":
    #     scheduler = ExponentialLRScheduler(
    #         optimizer, **optimizer_config["schedule"]["params"]
    #     )
    # elif scheduler_name == "poly":
    #     scheduler = PolyLR(optimizer, **optimizer_config["schedule"]["params"])
    # elif scheduler_name == "constant":
    #     scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    # elif scheduler_name == "linear":

    #     def linear_lr(it):
    #         return (
    #             it * optimizer_config["schedule"]["params"]["alpha"]
    #             + optimizer_config["schedule"]["params"]["beta"]
    #         )

    #     scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    if config.train.warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=int(
                config.train.warmup_epochs * num_samples
                / (num_gpus * train_bs)
            ),
            after_scheduler=scheduler,
        )

    return optimizer, scheduler
