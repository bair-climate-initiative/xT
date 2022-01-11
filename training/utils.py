import os
import pickle
import re

import cv2
import torch
from madgrad import MADGRAD
from timm.models import inception_v3
from timm.optim import AdamW
from torch import optim, nn
from torch.distributed import get_world_size
from torch.optim import lr_scheduler
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR, CosineAnnealingLR
from torch.optim.rmsprop import RMSprop

from training.schedulers import ExponentialLRScheduler, PolyLR, LRStepScheduler

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import torch.distributed as dist



def create_optimizer(optimizer_config, model, num_samples: int, num_gpus: int = 1):
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

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', "_bn0.weight", "_bn1.weight", "_bn2.weight"]

    def make_params(param_optimizer, lr=None):
        params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': optimizer_config["weight_decay"]},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        for p in params:
            if lr is not None:
                p["lr"] = lr
        return params

    if optimizer_config.get("classifier_lr", -1) != -1:
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

        params.extend(make_params(classifier_params, optimizer_config["classifier_lr"]))
        params.extend(make_params(net_params))
        print("param_groups", len(params))
    else:
        param_optimizer = list(model.named_parameters())
        params = make_params(param_optimizer)
        print("param_groups", len(params))
    train_bs = optimizer_config["train_bs"]
    epochs = optimizer_config["schedule"]["epochs"]
    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(params,
                              lr=optimizer_config["learning_rate"],
                              momentum=optimizer_config["momentum"],
                              nesterov=optimizer_config["nesterov"])

    elif optimizer_config["type"] == "Adam":
        optimizer = optim.Adam(params,
                               eps=optimizer_config.get("eps", 1e-8),
                               lr=optimizer_config["learning_rate"],
                               weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "AdamW":
        optimizer = AdamW(params,
                          eps=optimizer_config.get("eps", 1e-8),
                          lr=optimizer_config["learning_rate"],
                          weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "RmsProp":
        optimizer = RMSprop(params,
                            lr=optimizer_config["learning_rate"],
                            weight_decay=optimizer_config["weight_decay"])
    elif optimizer_config["type"] == "MadGrad":
        optimizer = MADGRAD(params,
                            lr=optimizer_config["learning_rate"])
    else:
        raise KeyError("unrecognized optimizer {}".format(optimizer_config["type"]))

    if optimizer_config["schedule"]["type"] == "step":
        scheduler = LRStepScheduler(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "cosine":
        tmax = int(epochs * num_samples/(num_gpus * train_bs))
        eta_min = optimizer_config["schedule"]["params"]["eta_min"]
        print(f"Cosine decay with T_max:{tmax} eta_min:{eta_min}")
        scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=eta_min)
    elif optimizer_config["schedule"]["type"] == "clr":
        scheduler = CyclicLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "multistep":
        scheduler = MultiStepLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "exponential":
        scheduler = ExponentialLRScheduler(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "poly":
        scheduler = PolyLR(optimizer, **optimizer_config["schedule"]["params"])
    elif optimizer_config["schedule"]["type"] == "constant":
        scheduler = lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif optimizer_config["schedule"]["type"] == "linear":
        def linear_lr(it):
            return it * optimizer_config["schedule"]["params"]["alpha"] + optimizer_config["schedule"]["params"]["beta"]

        scheduler = lr_scheduler.LambdaLR(optimizer, linear_lr)

    return optimizer, scheduler


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def load_checkpoint(model, checkpoint_path, strict=False):
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        state_dict = {re.sub("^module.", "", k): w for k, w in state_dict.items()}
        orig_state_dict = model.state_dict()
        mismatched_keys = []
        for k, v in state_dict.items():
            ori_size = orig_state_dict[k].size() if k in orig_state_dict else None
            if v.size() != ori_size:
                print("SKIPPING!!! Shape of {} changed from {} to {}".format(k, v.size(), ori_size))
                mismatched_keys.append(k)
        for k in mismatched_keys:
            del state_dict[k]
        model.load_state_dict(state_dict, strict=strict)
        del state_dict
        del orig_state_dict
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))
    else:
        model.load_state_dict(checkpoint)
    del checkpoint
