import datetime
import glob
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
import torch
import torch.distributed as dist
from einops import rearrange
from torch.cuda import empty_cache
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference.postprocessing import process_confidence
from inference.run_inference import predict_scene_and_return_mm
from metrics import xview_metric
from metrics.xview_metric import create_metric_arg_parser
from training.utils import (
    get_rank,
    is_dist_avail_and_initialized,
    is_main_process,
)

from .config import XviewConfig
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
    def get_improved_metrics(
        self, prev_metrics: Dict, current_metrics: Dict
    ) -> Dict:
        pass


class XviewEvaluator(Evaluator):
    def __init__(self, config: XviewConfig):
        super().__init__()
        self.config = config
        mode = "public" if config.test else "val"
        self.mode = mode

        self.crop_size = config.data.val_crop_size
        self.tiling = config.model.tiling
        self.input_size = config.model.backbone.img_size
        # self.patch_size = config.model.backbone.patch_size
        # self.context_patch_len = config.context_patch_len
        self.overlap = config.data.overlap
        if mode == "public":
            self.dataset_dir = "images/public"
            self.annotation_dir = "labels/public.csv"
            self.shoreline_dir = "shoreline/public"
        else:
            self.dataset_dir = "images/validation"
            self.annotation_dir = "labels/validation.csv"
            self.shoreline_dir = "shoreline/validation"

    def init_metrics(self) -> Dict:
        return {"xview": 0}

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
            yield batch_new, k, (
                self.input_size * i,
                self.input_size * (i + 1),
                self.input_size * j,
                self.input_size * (j + 1),
                batch.shape[-2],
                batch.shape[-1],
            ),{} 

    # def build_iterator(self, batch):
    #     old_dim = self.crop_size
    #     n = old_dim // self.input_size
    #     rearranged_image = rearrange(
    #         batch,
    #         "N C (H PH GH) (W PW GW )-> N C H W  PH PW GH GW",
    #         PH=self.input_size // self.patch_size,
    #         PW=self.input_size // self.patch_size,
    #         GH=self.patch_size,
    #         GW=self.patch_size,
    #     )
    #     N, C, H, W, PH, PW, PPH, PPW = rearranged_image.shape
    #     rearranged_image = rearranged_image.flatten(2, 5)
    #     for i, j, k in build_tiling(n, self.tiling):
    #         indices = torch.rand(N, H, W, PH, PW)
    #         indices[:, i, j] = 999
    #         indices = indices.flatten(1).argsort(-1)
    #         indices = indices[:, : self.context_patch_len]
    #         context_patches = torch.stack(
    #             [rearranged_image[i][:, v] for i, v in enumerate(indices)],
    #             dim=0,
    #         )  # N C L 16 16
    #         H_i = indices // (W * PH * PW)
    #         W_i = (indices // (PH * PW)) % W
    #         PH_i = (indices // (PW)) % PH
    #         PW_i = indices % PW
    #         # assert torch.all(indices == H_i * (W * PH*PW) + W_i *PH*PW + PH_i * PW + PW_i) sanity check
    #         h_idx = H_i * PH + PH_i
    #         w_idx = W_i * PW + PW_i

    #         raw_indices_h = torch.arange(PH) + i * PH
    #         raw_indices_w = torch.arange(PH) + j * PW
    #         raw_indices = torch.stack(
    #             [
    #                 raw_indices_h[:, None].repeat(1, PW),
    #                 raw_indices_w[None,].repeat(PH, 1),
    #             ]
    #         )
    #         patch_indices = torch.stack([h_idx, w_idx])  # 2 X B X L

    #         batch_new = batch[
    #             ...,
    #             self.input_size * i : self.input_size * (i + 1),
    #             self.input_size * j : self.input_size * (j + 1),
    #         ]
    #         context_id = i * n + j
    #         context = {}
    #         context["context_patches"] = context_patches
    #         context["patch_indices"] = patch_indices
    #         context["raw_indices"] = raw_indices
    #         yield batch_new, k, (
    #             self.input_size * i,
    #             self.input_size * (i + 1),
    #             self.input_size * j,
    #             self.input_size * (j + 1),
    #             batch.shape[-2],
    #             batch.shape[-1],
    #         ), context

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        *args,**kwargs,
    ) -> Dict:
        if is_main_process():
            print("DEBUG: MAIN")
        conf_name = os.path.splitext(os.path.basename(self.config.name))[0]
        val_dir = os.path.join(
            self.config.output_dir, conf_name, str(self.config.data.fold)
        )
        os.makedirs(val_dir, exist_ok=True)
        dataset_dir = os.path.join(self.config.data.dir, self.dataset_dir)
        extra_context = True # always set to True, this flag just mean we will perform the check on each
        if is_main_process() and self.config.train.test_reset:
            csv_paths = glob.glob(os.path.join(val_dir, "*.csv"))
            for csv_file in csv_paths:
                os.remove(csv_file)
        if is_dist_avail_and_initialized():
            dist.barrier()
        rank = get_rank()
        for sample in tqdm(dataloader, position=0):
            scene_id = sample["name"][0]
            tgt_path = os.path.join(val_dir, f"{scene_id}.csv")
            logging.debug(f"{rank}:Evaluating {scene_id} ")
            if (
                self.config.test
                and os.path.exists(tgt_path)
                and datetime.datetime.fromtimestamp(os.path.getmtime(tgt_path))
                > datetime.datetime.now() - datetime.timedelta(hours=10)
            ):
                continue
            mask_dict = predict_scene_and_return_mm(
                [model],
                scene_id=scene_id,
                dataset_dir=dataset_dir,
                use_fp16=self.config.fp16,
                rotate=True,
                crop_size=self.crop_size,
                overlap=self.overlap,
                extra_context=extra_context,
                iter_function=self.build_iterator,
                position=get_rank() + 1,
            )
            data = process_confidence(scene_id, None, mask_dict)
            pd.DataFrame(
                data,
                columns=[
                    "detect_scene_row",
                    "detect_scene_column",
                    "scene_id",
                    "is_vessel",
                    "is_fishing",
                    "vessel_length_m",
                    "confidence",
                    "mean_obj",
                    "mean_vessel",
                    "mean_fishing",
                    "mean_length",
                    "mean_center",
                ],
            ).to_csv(os.path.join(val_dir, f"{scene_id}.csv"))
        if is_dist_avail_and_initialized():
            dist.barrier()
        xview = 0
        output = {}
        if is_main_process():
            csv_paths = glob.glob(os.path.join(val_dir, "*.csv"))
            pred_csv = f"pred_{conf_name}_{self.config.data.fold}.csv"
            print(csv_paths)
            pd.concat(
                [pd.read_csv(csv_path).reset_index() for csv_path in csv_paths]
            ).to_csv(pred_csv, index=False)
            parser = create_metric_arg_parser()
            metric_args = parser.parse_args("")
            df = pd.read_csv(pred_csv)
            df = df.reset_index()
            df[
                [
                    "detect_scene_row",
                    "detect_scene_column",
                    "scene_id",
                    "is_vessel",
                    "is_fishing",
                    "vessel_length_m",
                ]
            ].to_csv(pred_csv, index=False)
            metric_args.inference_file = pred_csv
            metric_args.label_file = os.path.join(
                self.config.data.dir, self.annotation_dir
            )
            metric_args.shore_root = os.path.join(
                self.config.data.dir, self.shoreline_dir
            )
            metric_args.shore_tolerance = 2
            metric_args.costly_dist = True
            metric_args.drop_low_detect = True
            metric_args.distance_tolerance = 200
            metric_args.output = os.path.join(self.config.log_dir, "out.json")
            output = xview_metric.evaluate_xview_metric(metric_args)
            xview = output["aggregate"]
        if is_dist_avail_and_initialized():
            dist.barrier()
        empty_cache()
        return {"xview": xview, **output}

    def get_improved_metrics(
        self, prev_metrics: Dict, current_metrics: Dict
    ) -> Dict:
        improved = {}
        for k in ("xview", "loc_fscore_shore"):
            if current_metrics[k] > prev_metrics.get(k, 0.0):
                print(
                    k,
                    " improved from {:.4f} to {:.4f}".format(
                        prev_metrics["xview"], current_metrics["xview"]
                    ),
                )
                improved[k] = current_metrics[k]
        return improved
