from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.distributed as dist
from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall
from tqdm import tqdm

from xt.utils import is_dist_avail_and_initialized, is_main_process

from .tiling import build_tiling


class Evaluator(ABC):
    @abstractmethod
    def init_metrics(self) -> Dict:
        pass

    @abstractmethod
    def validate(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
    ) -> Dict:
        pass

    @abstractmethod
    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        pass


class ClsEvaluator(Evaluator):
    def __init__(self, config):
        super().__init__()
        self.config = config
        mode = "public" if config.test else "val"
        self.mode = mode

        self.crop_size = config.data.val_crop_size
        self.tiling = config.model.tiling
        self.input_size = config.model.backbone.img_size
        # self.patch_size = config.model.backbone.patch_size
        # self.context_patch_len = config.context_patch_len
        self.num_classes = config.model.num_classes

        self.top1_acc = Accuracy(
            task="multiclass", num_classes=self.num_classes, top_k=1
        ).cuda()
        self.top5_acc = Accuracy(
            task="multiclass", num_classes=self.num_classes, top_k=5
        ).cuda()
        self.precision = Precision(
            task="multiclass", average="macro", num_classes=self.num_classes
        ).cuda()
        self.recall = Recall(
            task="multiclass", average="macro", num_classes=self.num_classes
        ).cuda()

    def init_metrics(self) -> Dict:
        return {"accuracy_top1": 0, "accuracy_top5": 0, "precision": 0, "recall": 0}

    def build_iterator(self, batch):
        old_dim = self.crop_size
        n = old_dim // self.input_size
        for i, j, k in build_tiling(n, self.tiling):
            batch_new = batch[
                ...,
                self.input_size * i : self.input_size * (i + 1),
                self.input_size * j : self.input_size * (j + 1),
            ]
            context_id = i * n + j
            yield (
                batch_new,
                k,
                (
                    self.input_size * i,
                    self.input_size * (i + 1),
                    self.input_size * j,
                    self.input_size * (j + 1),
                    batch.shape[-2],
                    batch.shape[-1],
                ),
                {},
                (i, j),
            )

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        *args,
        **kwargs,
    ) -> Dict:
        # Torchmetrics reset
        for metric in [self.top1_acc, self.top5_acc, self.precision, self.recall]:
            metric.reset()

        def model_foward(x, model):
            mem = set()
            output = []
            iterator = self.build_iterator(x)
            for batch_new, k, (x0, x1, y0, y1, hh, ww), context, cord in iterator:
                mem_only = k.get("mem_only", False)
                local_output, mem = model(
                    batch_new, context=context, cord=cord, mem=mem
                )
                if mem_only:
                    continue
                # context_id = k["context_id"]
                output.append(local_output)
            final_out = {}
            for k, v in output[0].items():
                final_out[k] = torch.stack([z[k] for z in output])
            final_out["label"] = final_out["label"][-1]
            return final_out

        metrics = {}

        if is_dist_avail_and_initialized():
            dist.barrier()

        dataloader_tqdm = tqdm(dataloader, position=0)
        dataloader = iter(dataloader)

        for _ in range(len(dataloader)):
            sample = next(dataloader)
            img = sample["image"].float()
            output = model(img)
            pred = output["label"]
            gt = sample["label"]

            # Torchmetrics update
            for metric in [self.top1_acc, self.top5_acc, self.precision, self.recall]:
                metric.update(pred.cuda().softmax(dim=-1), gt.cuda())

            if is_main_process():
                dataloader_tqdm.update()

        top1_acc_tm = self.top1_acc.compute()
        top5_acc_tm = self.top5_acc.compute()
        precion_tm = self.precision.compute()
        recall_tm = self.recall.compute()

        if is_main_process():
            metrics = {
                "accuracy_top1": top1_acc_tm.item(),
                "accuracy_top5": top5_acc_tm.item(),
                "precision": precion_tm.item(),
                "recall": recall_tm.item(),
            }
            dataloader_tqdm.set_postfix({**metrics})

        if is_dist_avail_and_initialized():
            dist.barrier()
        empty_cache()

        return metrics

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        for k in ["accuracy_top1", "accuracy_top5", "precision", "recall"]:
            if current_metrics[k] > prev_metrics.get(k, 0.0):
                print(
                    k,
                    " improved from {:.4f} to {:.4f}".format(
                        prev_metrics[k], current_metrics[k]
                    ),
                )
                improved[k] = current_metrics[k]
        return improved


def build_evaluator(cfg):
    if cfg.data.dataset == "inaturalist":
        cls = ClsEvaluator
    return cls(cfg)
