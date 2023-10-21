import logging
import math
import os
import re
import time
from dataclasses import dataclass
from numbers import Number
from typing import Any, Dict, List

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
import wandb
import yaml
from einops import rearrange
from fvcore.nn import FlopCountAnalysis
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from timm.utils import AverageMeter
from torch.nn import DataParallel, SyncBatchNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import zoo
from train_val_segmentor import XviewConfig
from training import losses
from training.evaluator import Evaluator
from training.losses import LossCalculator
from training.optimizer import create_optimizer
from training.sampler import DistributedWeightedRandomSampler
from training.utils import (
    SmoothedValue,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    wandb_dump_images,
)

from .tiling import build_tiling

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO").upper())


@dataclass
class TrainConfig:
    """Full training config."""

    epochs: int = 120
    """Number of epochs to train for."""
    batch_size: int = 4
    """Batch size per GPU."""
    base_lr: float = 1e-3
    """Base learning rate (adjusted by effective batch size)."""
    min_lr_multiplier: float = 0.01
    """Minimum learning rate to anneal to as a factor of base_lr."""
    warmup_lr_multiplier: float = 0.001
    """Warmup learning rate to start at as a factor of base_lr."""
    classifier_multiplier: float = 1.0
    """Multiplier for classifier learning rate."""
    warmup_epochs: int = 0
    """Number of epochs to warmup for."""
    weight_decay: float = 1e-4
    """Weight decay."""
    val_batch_size: int = 2
    """Validation batch size per GPU."""
    freeze_epochs: int = 0
    """Number of epochs to freeze encoder for."""
    positive_ratio: float = 0.85
    """Ratio of positive samples in a batch."""


class LossFunction:
    def __init__(
        self,
        loss: LossCalculator,
        name: str,
        weight: float = 1,
        display: bool = False,
    ):
        super().__init__()
        self.loss = loss
        self.name = name
        self.weight = weight
        self.display = display


class PytorchTrainer:
    def __init__(
        self,
        config: XviewConfig,
        evaluator: Evaluator,
        train_data: Dataset,
        val_data: Dataset,
    ) -> None:
        super().__init__()
        self.config = config

        self.wandb_id = None
        if is_main_process():
            # TODO*: Investigate how to properly dump config.
            wandb_args = dict(
                project="xview3 detection unet",
                entity="bair-climate-initiative",
                resume="allow",
                name=config.name,
                config=config,
                dir=config.log_dir,
            )
            artifact = wandb.Artifact(
                "config_file",
                type="config_file",
            )
            config_dump_path = os.path.join(
                self.config.output_dir, "config.yaml"
            )
            with open(config_dump_path, "w") as outfile:
                outfile.write(OmegaConf.to_yaml(self.config))
            artifact.add_file(config_dump_path)

            wandb.init(**wandb_args)
            wandb.log_artifact(artifact)
            self.wandb_id = wandb.run.id

        self._init_distributed()
        self.evaluator = evaluator
        self.current_metrics = evaluator.init_metrics()
        self.current_epoch = 0
        self.model = self._init_model()
        # TODO: how tf to pass in T-XL? tell shufan to do maybe
        self.patch_size = self.config.patch_size
        self.context_patch_len = self.config.context_patch_len

        self.losses = self._init_loss_functions()
        self.scale_learning_rate()

        self._init_amp()
        self.optimizer, self.scheduler = create_optimizer(
            self.config.optimizer,
            self.model,
            len(train_data),
        )
        self.train_data = train_data
        self.val_data = val_data
        if is_main_process():
            print(self.model)
            self.summary_writer = SummaryWriter(
                os.path.join(config.log_dir, self.snapshot_name)
            )

        # self._profile_model((1, 2, self.conf["crop_size"], self.conf["crop_size"]))

    def validate(self, test_loader=None):
        if self.config.distributed:
            dist.barrier()
        self.model.eval()
        metrics = self.evaluator.validate(
            test_loader if test_loader is not None else self.get_val_loader(),
            self.model,
            distributed=self.config.distributed,
            local_rank=self.config.local_rank,
            snapshot_name=self.snapshot_name,
        )
        print(metrics)
        if self.config.local_rank == 0 and wandb.run is not None:
            wandb.log(metrics)

    def fit(self):
        for epoch in range(
            self.current_epoch, self.conf["optimizer"]["schedule"]["epochs"]
        ):
            rank = self.config.local_rank
            logging.debug(f"{ rank}: epoch start")
            if self.config.distributed:
                dist.barrier()
            self.current_epoch = epoch
            logging.debug(f"{ rank}: set train mode")
            self.model.train()
            self._freeze()
            self._run_one_epoch_train(self.get_train_loader())
            torch.cuda.synchronize()
            if self.config.distributed:
                dist.barrier()
            self.model.eval()
            self._save_last()
            logging.debug(f"{rank} Epoch finished")
            if (self.current_epoch + 1) % self.config.test_every == 0:
                logging.debug(f"{rank} eval launched")
                metrics = self.evaluator.validate(
                    self.get_val_loader(),
                    self.model,
                    distributed=self.config.distributed,
                    local_rank=self.config.local_rank,
                    snapshot_name=self.snapshot_name,
                )
                logging.debug(f"{rank} eval done")
                if self.config.local_rank == 0 and wandb.run is not None:
                    metrics["epoch"] = epoch
                    wandb.log(metrics)
                improved_metrics = None
                if self.config.local_rank == 0:
                    improved_metrics = self.evaluator.get_improved_metrics(
                        self.current_metrics, metrics
                    )
                    self.current_metrics.update(improved_metrics)
                self._save_best(improved_metrics)
                if self.config.local_rank == 0:
                    for k, v in metrics.items():
                        self.summary_writer.add_scalar(
                            "val/{}".format(k),
                            float(v),
                            global_step=self.current_epoch,
                        )

    def scale_learning_rate(self):
        # linear scale the learning rate according to total batch size, may not be optimal
        config = self.conf["optimizer"]
        eff_batch_size = config["train_bs"] * dist.get_world_size()
        # base batch size is 8 * 1 = 32
        lr = config["learning_rate"] * eff_batch_size / 8.0
        classifier_lr = config["classifier_lr"] * eff_batch_size / 8.0
        if self.config.local_rank == 0:
            print(
                f"Effective batch size of {eff_batch_size} "
                f"lr: {config['learning_rate']} --> {lr}"
            )
        config["learning_rate"] = lr
        config["classifier_lr"] = classifier_lr
        self.conf["optimizer"] = config

    def _profile_model(self, shape):
        input = torch.randn(shape).cuda()
        flops = FlopCountAnalysis(self.model, input)
        r = flops.by_operator()
        print(r)
        print(dict(total_flops=sum(r.values())))

    def _get_state_dict(self):
        state_dict = self.model.state_dict()
        return state_dict
        # return {k:v.cpu() for k,v in state_dict.items()}

    def _save_last(self):
        # self.model = self.model.eval()
        payload = {
            "epoch": self.current_epoch,
            "state_dict": self._get_state_dict(),
            "metrics": self.current_metrics,
        }
        if self.config.local_rank == 0:
            torch.save(
                payload,
                os.path.join(
                    self.config.output_dir,
                    self.snapshot_name + "_" + str(self.wandb_id) + "_" "_last",
                ),
            )

    def _save_best(self, improved_metrics: Dict):
        payload = {
            "epoch": self.current_epoch,
            "state_dict": self._get_state_dict(),
            "metrics": self.current_metrics,
        }
        if self.config.local_rank == 0:
            for metric_name in improved_metrics.keys():
                torch.save(
                    payload,
                    os.path.join(
                        self.config.output_dir,
                        self.snapshot_name
                        + "_"
                        + str(self.wandb_id)
                        + "_"
                        + metric_name,
                    ),
                )

    def build_iterator(self, dataloader):
        for x in dataloader:
            old_dim = x["image"].shape[-1]
            n = old_dim // self.input_size
            rearranged_image = rearrange(
                x["image"],
                "N C (H PH GH) (W PW GW )-> N C H W  PH PW GH GW",
                PH=self.input_size // self.patch_size,
                PW=self.input_size // self.patch_size,
                GH=self.patch_size,
                GW=self.patch_size,
            )
            N, C, H, W, PH, PW, PPH, PPW = rearranged_image.shape
            rearranged_image = rearranged_image.flatten(2, 5)
            for i, j, k in build_tiling(n, self.config.model.tiling):
                indices = torch.rand(N, H, W, PH, PW)
                indices[:, i, j] = 999
                indices = indices.flatten(1).argsort(-1)
                indices = indices[:, : self.context_patch_len]
                context_patches = torch.stack(
                    [rearranged_image[i][:, v] for i, v in enumerate(indices)],
                    dim=0,
                )  # N C L 16 16
                H_i = indices // (W * PH * PW)
                W_i = (indices // (PH * PW)) % W
                PH_i = (indices // (PW)) % PH
                PW_i = indices % PW
                # assert torch.all(indices == H_i * (W * PH*PW) + W_i *PH*PW + PH_i * PW + PW_i) sanity check
                h_idx = H_i * PH + PH_i
                w_idx = W_i * PW + PW_i

                raw_indices_h = torch.arange(PH) + i * PH
                raw_indices_w = torch.arange(PH) + j * PW
                raw_indices = torch.stack(
                    [
                        raw_indices_h[:, None].repeat(1, PW),
                        raw_indices_w[None,].repeat(PH, 1),
                    ]
                )
                patch_indices = torch.stack([h_idx, w_idx])  # 2 X B X L
                new_payload = {
                    k: v[
                        ...,
                        self.input_size * i : self.input_size * (i + 1),
                        self.input_size * j : self.input_size * (j + 1),
                    ]
                    for k, v in x.items()
                    if k != "name"
                }
                new_payload["context_patches"] = context_patches
                new_payload["patch_indices"] = patch_indices
                new_payload["raw_indices"] = raw_indices
                new_payload["name"] = x["name"]
                new_payload["context_id"] = k["context_id"]
                new_payload["mem_only"] = k.get("mem_only", False)
                yield new_payload

    def _run_one_epoch_train(self, loader: DataLoader):
        torch.autograd.set_detect_anomaly(True)
        iterator = loader
        # data_time = SmoothedValue(fmt="{avg:.4f}")
        loss_meter = AverageMeter()
        # forward_time = SmoothedValue(fmt="{avg:.4f}")
        # backward_time = SmoothedValue(fmt="{avg:.4f}")
        avg_meters = {"loss": loss_meter}
        for loss_def in self.losses:
            if loss_def.display:
                avg_meters[loss_def.name] = AverageMeter()

        if self.conf["optimizer"]["schedule"]["mode"] == "epoch":
            self.scheduler.step(self.current_epoch)
        extra_context = self.model.module.extra_context
        if extra_context:
            iterator = self.build_iterator(iterator)
            iter_scale = (self.conf["crop_size"] // self.input_size) ** 2
            if "two_stream" in self.tiling:
                iter_scale *= 2
        else:
            iter_scale = 1
        #         total_n = iter_scale * len(loader)
        #         if self.config.local_rank == 0:
        #             t = tqdm(total=total_n)
        #         loader = iter(loader)

        #         for i in range(total_n):
        #             end = time.time()
        #             sample = next(loader)
        #             data_time.update(time.time() - end)
        #             if self.config.local_rank == 0:
        #                 t.update()
        # Sliced Images with context_id
        # todo: make configurable
        if self.config.local_rank == 0:
            iterator = tqdm(iterator, total=iter_scale * len(loader))
        # Broken, temporaily disable time logging
        for i, sample in enumerate(iterator):
            # if i > 2:
            #     break
            imgs = sample["image"].cuda().float()
            if extra_context:
                if sample["context_id"] == 0:
                    mem = tuple()
                else:
                    pass
                context = dict(
                    context_patches=sample["context_patches"].cuda(),
                    patch_indices=sample["patch_indices"],
                    raw_indices=sample["raw_indices"],
                )
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
                with torch.autograd.detect_anomaly():
                    if extra_context and sample["mem_only"]:
                        with torch.no_grad():
                            output, mem = self.model(
                                imgs, context=context, mem=mem
                            )
                        continue
                    elif extra_context:
                        output, mem = self.model(imgs, context=context, mem=mem)
                    else:
                        output = self.model(imgs)
                    # forward_time.update(time.time() - end)
                    # if i % 400 == 0:
                    # visualize
                    # if self.config.local_rank == 0:
                    #     all_keys = []
                    #     all_imgs = [imgs[0][0].detach().cpu()]
                    #     all_keys.append('input')
                    #     for k,v in output.items():
                    #         all_imgs.append(v[0][0].detach().cpu())
                    #         all_imgs.append(sample[k][0][0].detach().cpu())
                    #         short_k = k.replace('_mask','')
                    #         all_keys.append(k)
                    #         all_keys.append(k+'_GT')

                    #     wandb_dump_images(all_imgs,keys=all_keys)
                    total_loss = 0
                    for loss_def in self.losses:
                        l = loss_def.loss.calculate_loss(output, sample)
                        if math.isnan(l.item()) or math.isinf(l.item()):
                            print(loss_def)
                            print("is nan!")
                        if loss_def.display:
                            avg_meters[loss_def.name].update(
                                l if isinstance(l, Number) else l.item(),
                                imgs.size(0),
                            )
                        total_loss += loss_def.weight * l
            loss_meter.update(total_loss.item(), imgs.size(0))
            if math.isnan(total_loss.item()) or math.isinf(total_loss.item()):
                raise ValueError("NaN loss !!")
            avg_metrics = {k: f"{v.avg:.4f}" for k, v in avg_meters.items()}
            if (
                self.config.local_rank == 0
                and wandb.run is not None
                and i % 50 == 0
            ):
                payload = {
                    k: float(f"{v.avg:.4f}") for k, v in avg_meters.items()
                }
                payload.update(dict(lr=float(self.scheduler.get_lr()[-1])))
                wandb.log(payload)

            # Run backward pass
            # end = time.time()
            # print(total_loss.dtype,'hh')
            if (
                self.config.fp16 and False
            ):  # sth happened here, really need to check whats wrong
                print(total_loss.device, "hh")
                # self.gscaler.scale(total_loss).backward()
                # self.gscaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                # self.gscaler.step(self.optimizer)
                # self.gscaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
            # backward_time.update(time.time() - end)

            torch.cuda.synchronize()
            if self.config.distributed:
                dist.barrier()
            if self.conf["optimizer"]["schedule"]["mode"] in ("step", "poly"):
                self.scheduler.step(
                    int(i / iter_scale) + self.current_epoch * len(loader)
                )
            if self.config.local_rank == 0:
                iterator.set_postfix(
                    {
                        "lr": float(self.scheduler.get_lr()[-1]),
                        "epoch": self.current_epoch,
                        "mem": f"{torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}G",
                        **avg_metrics,
                    }
                )
        if self.config.local_rank == 0:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                lr = param_group["lr"]
                self.summary_writer.add_scalar(
                    "group{}/lr".format(idx),
                    float(lr),
                    global_step=self.current_epoch,
                )
            self.summary_writer.add_scalar(
                "train/loss",
                float(loss_meter.avg),
                global_step=self.current_epoch,
            )

    @property
    def train_batch_size(self):
        return self.config.train.batch_size

    @property
    def val_batch_size(self):
        return self.config.train.val_batch_size

    def get_train_loader(self) -> DataLoader:
        train_sampler = None
        if is_dist_avail_and_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_data
            )
            if hasattr(self.train_data, "get_weights"):
                train_sampler = DistributedWeightedRandomSampler(
                    self.train_data, self.train_data.get_weights()
                )
            train_sampler.set_epoch(self.current_epoch)
        train_data_loader = DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            pin_memory=False,
            drop_last=True,
        )

        return train_data_loader

    def get_val_loader(self) -> DataLoader:
        val_sampler = None
        if is_dist_avail_and_initialized():
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                self.val_data,
                shuffle=False,
                num_replicas=get_world_size(),
                rank=get_rank(),
            )
        val_data_loader = DataLoader(
            self.val_data,
            sampler=val_sampler,
            batch_size=self.val_batch_size,
            num_workers=self.config.data.num_workers,
            shuffle=False,
            pin_memory=False,
        )
        return val_data_loader

    @property
    def snapshot_name(self):
        return "{}{}_{}_{}".format(
            self.config.model.name,
            self.config.model.encoder.name,
            self.config.data.fold,
        )

    def _freeze(self):
        if hasattr(self.model.module, "encoder"):
            encoder = self.model.module.encoder
        elif hasattr(self.model.module, "encoder_stages"):
            encoder = self.model.module.encoder_stages
        else:
            logging.warn("unknown encoder model")
            return
        if self.current_epoch < self.config.train.freeze_epochs:
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
        else:
            encoder.train()
            for p in encoder.parameters():
                p.requires_grad = True
        if self.config.freeze_bn:
            for m in self.model.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def _init_amp(self):
        self.gscaler = torch.cuda.amp.GradScaler()

        if self.config.distributed and self.config.fsdp:
            from torch.distributed.fsdp import (
                CPUOffload,
                FullyShardedDataParallel,
                MixedPrecision,
            )
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            self.model = FullyShardedDataParallel(
                self.model,
                auto_wrap_policy=size_based_auto_wrap_policy,
                # sharding_strategy=size_based_auto_wrap_policy,
                # fsdp_auto_wrap_policy=default_auto_wrap_policy,
                cpu_offload=CPUOffload(offload_params=True),
                # mixed_precision=MixedPrecision(param_dtype=torch.float16)
                # device_id=self.config.local_rank,
                # output_device=self.config.local_rank,
                # find_unused_parameters=False,
            )
        elif self.config.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[get_rank()],
                output_device=get_rank(),
                find_unused_parameters=False,
            )
        else:
            self.model = DataParallel(self.model).cuda()

    def _init_distributed(self):
        # TODO!: Make sure this is initialized correctly.
        if self.config.distributed:
            self.pg = dist.init_process_group(
                backend="nccl",
                # rank=self.config.local_rank, set to torchrun
                # world_size=self.config.world_size,
            )

            torch.cuda.set_device(get_rank())
        else:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            # os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu

    def _load_checkpoint(self, model: torch.nn.Module):
        checkpoint_path = self.config.model.resume
        if not checkpoint_path:
            return
        if os.path.isfile(checkpoint_path):
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
                        orig_state_dict[k].size()
                        if k in orig_state_dict
                        else None
                    )
                    if v.size() != ori_size:
                        print(
                            "SKIPPING!!! Shape of {} changed from {} to {}".format(
                                k, v.size(), ori_size
                            )
                        )
                        mismatched_keys.append(k)
                for k in mismatched_keys:
                    del state_dict[k]
                model.load_state_dict(state_dict, strict=False)
                # if not self.config.from_zero:
                #     self.current_epoch = checkpoint["epoch"]
                #     if not self.config.zero_score:
                #         self.current_metrics = checkpoint.get(
                #             "metrics", self.evaluator.init_metrics()
                #         )
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        checkpoint_path, checkpoint["epoch"]
                    )
                )
            else:
                model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_path))
        # if self.config.from_zero:
        #     self.current_metrics = self.evaluator.init_metrics()
        #     self.current_epoch = 0

    def _init_model(self):
        print(self.config)
        self.input_size = self.config.data.crop_size
        # TODO: please for the love of god fix this into a model builder
        model = zoo.__dict__[self.config.model.name](
            **self.config.encoder, crop_size=self.input_size
        )
        model = model.cuda()
        self._load_checkpoint(model)

        if self.config.distributed and not self.config.freeze_bn:
            model = SyncBatchNorm.convert_sync_batchnorm(model, self.pg)
        channels_last = self.config.model.encoder.channel_last
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        return model

    def _init_loss_functions(self) -> List[LossFunction]:
        assert self.conf["losses"]
        loss_functions = []
        for loss_def in self.conf["losses"]:
            # TODO: this one too, loss builder
            loss_fn = losses.__dict__[loss_def["type"]](**loss_def["params"])
            loss_weight = loss_def["weight"]
            display = loss_def["display"]
            loss_functions.append(
                LossFunction(loss_fn, loss_def["name"], loss_weight, display)
            )

        return loss_functions
