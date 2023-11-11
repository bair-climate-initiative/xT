from dataclasses import dataclass

# from hydra.core.config_store import ConfigStore
from timm.optim import AdamW
from torch import nn, optim
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler

from .schedulers import LRStepScheduler
from .utils import get_world_size, is_main_process


@dataclass
class OptimizerConfig:
    """Optimizer Configuation: name, lr + options, warmup, wd."""

    name: str = "sgd"
    """Optimizer shortname: options are [sgd, adam, adamw]"""
    lr: float = 1e-3
    """Absolute learning rate, NOT SET DIRECTLY! Overridden below."""
    base_lr: float = 1e-3
    """Base learning rate (adjusted by effective batch size)."""
    min_lr_ratio: float = 0.01
    """Minimum learning rate to anneal to as a factor of base_lr."""
    classifier_ratio: float = 1.0
    """Multiplier for classifier learning rate."""
    warmup_epochs: int = 0
    """Number of epochs to warmup for."""
    weight_decay: float = 1e-4
    """Weight decay."""
    scheduler: str = "cosine"
    """Learning rate scheduler."""
    mode: str = "step"
    """Scheduler mode: [epoch, step, poly]."""
    momentum: float = 0.9
    """SGD Momentum"""
    nesterov: bool = True
    """SGD Nesterov momentum"""


# @dataclass
# class SGDConfig(OptimizerConfig):
#     """SGD Configuration: momentum, nesterov."""

#     momentum: float = 0.9
#     """SGD Momentum"""
#     nesterov: bool = True
#     """SGD Nesterov momentum"""


# @dataclass
# class AdamConfig(OptimizerConfig):
#     """Adam Configuration: eps, beta."""

#     eps: float = 1e-8
#     """Adam epsilon."""
#     betas: Tuple[float, float] = (0.9, 0.999)
#     """Adam betas"""


# @dataclass
# class AdamWConfig(OptimizerConfig):
#     """AdamW Configuration: eps, beta."""

#     eps: float = 1e-8
#     """AdamW epsilon."""
#     betas: Tuple[float, float] = (0.9, 0.999)
#     """AdamW betas"""


# cs = ConfigStore.instance()
# cs.store(group="optimizer", name="optimizer", node=OptimizerConfig)
# cs.store(group="optimizer", name="sgd", node=SGDConfig)
# cs.store(group="optimizer", name="adam", node=AdamConfig)
# cs.store(group="optimizer", name="adamw", node=AdamWConfig)


def create_optimizer(
    config: OptimizerConfig,
    model: nn.Module,
    loader_len: int,
    epochs: int,
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
                "weight_decay": config.weight_decay,
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

    if config.classifier_ratio != 1.0:
        classifier_lr = config.lr * config.classifier_ratio
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
        if is_main_process():
            print("param_groups", len(params))
    else:
        param_optimizer = list(model.named_parameters())
        params = make_params(param_optimizer)
        if is_main_process():
            print("param_groups", len(params))
    optimizer_name = str.lower(config.name)
    if optimizer_name == "sgd":
        optimizer = optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    elif optimizer_name == "adam":
        optimizer = optim.Adam(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_name))

    scheduler_name = config.scheduler
    eta_min = config.lr * config.min_lr_ratio
    if scheduler_name == "step":
        scheduler = LRStepScheduler(optimizer, eta_min=eta_min)
    elif scheduler_name == "cosine":
        tmax = int(epochs * loader_len)
        if is_main_process():
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

    if config.warmup_epochs > 0:
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=int(config.warmup_epochs * loader_len),
            after_scheduler=scheduler,
        )

    return optimizer, scheduler
