import os
import pickle
import re
from collections import deque

import cv2
import numpy as np
import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from torch.utils.data import Subset
import functools
import wandb

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

class ConflictResolver:
    """
    Utility methods for dealing with dictionaries.
    """
    @staticmethod
    def to_object(item):
        """
        Convert a dictionary to an object (recursive).
        """
        def convert(item): 
            if isinstance(item, dict):
                return type('jo', (), {k: convert(v) for k, v in item.items()})
            if isinstance(item, list):
                def yield_convert(item):
                    for index, value in enumerate(item):
                        yield convert(value)
                return list(yield_convert(item))
            else:
                return item
        return convert(item)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


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
        tensor_list.append(
            torch.empty((max_size,), dtype=torch.uint8, device="cuda")
        )
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def load_checkpoint(model, checkpoint_path, strict=False, verbose=True):
    if verbose:
        print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = {
            re.sub("^module.", "", k): w for k, w in state_dict.items()
        }
        orig_state_dict = model.state_dict()
        mismatched_keys = []
        for k, v in state_dict.items():
            ori_size = (
                orig_state_dict[k].size() if k in orig_state_dict else None
            )
            if v.size() != ori_size:
                if verbose:
                    print(
                        "SKIPPING!!! Shape of {} changed from {} to {}".format(
                            k, v.size(), ori_size
                        )
                    )
                mismatched_keys.append(k)
        for k in mismatched_keys:
            del state_dict[k]
        model.load_state_dict(state_dict, strict=strict)
        del state_dict
        del orig_state_dict
        print(
            "=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_path, checkpoint["epoch"]
            )
        )
    else:
        model.load_state_dict(checkpoint)
    del checkpoint


def get_random_subset(dataset, count=5):
    # n = len(dataset)
    indices = np.arange(count).tolist()
    # indices = np.random.choice(indices,count,replace=False)
    return Subset(dataset, indices)


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device="cuda"
        )
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def wandb_dump_images(imgs, name="vis", keys=None, epoch=0):
    """
    x: H X W X C
    y: H X W X C
    """
    if wandb.run is not None:
        n_imgs = len(imgs)
        fig, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
        for idx, img in enumerate(imgs):
            if torch.is_tensor(imgs):
                imgs = imgs.detach().cpu().numpy()
            axes[idx].imshow(img)
            if keys:
                axes[idx].title.set_text(keys[idx])
        fig.tight_layout()
        wandb.log({name: wandb.Image(fig), "epoch": epoch})
        plt.close(fig)



@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD
def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output