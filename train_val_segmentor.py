import glob
import os
import warnings

import torch

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

from torch.cuda import empty_cache

torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 120
import datetime

import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm

from inference.postprocessing import process_confidence
from inference.run_inference import predict_scene_and_return_mm
from metrics import xview_metric
from metrics.xview_metric import create_metric_arg_parser
from training.config import load_config
from training.val_dataset import XviewValDataset


class TestDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.names = os.listdir(root_dir)

    def __getitem__(self, index):
        return dict(name=self.names[index])

    def __len__(self):
        return len(self.names)


warnings.filterwarnings("ignore")
import argparse
import os
from typing import Dict

import pandas as pd
import torch.distributed
from torch.utils.data import DataLoader

from training.config import load_config
from training.tiling import build_tiling
from training.trainer import Evaluator, PytorchTrainer, TrainConfiguration


class XviewEvaluator(Evaluator):
    def __init__(self, args, mode="val") -> None:
        super().__init__()
        self.args = args
        if args.test:
            mode = "public"
        self.crop_size = args.crop_size_val
        self.conf = load_config(args.config, args=args)
        self.tiling = self.conf.get("tiling", "naive")
        self.input_size = self.conf.get("encoder_crop_size", self.conf["crop_size"])
        self.overlap = args.overlap_val
        if mode == "public":
            self.dataset_dir = "images/public"
            self.annotation_dir = "labels/public.csv"
            self.shoreline_dir = "shoreline/public"
        elif mode == "val":
            self.dataset_dir = "images/validation"
            self.annotation_dir = "labels/validation.csv"
            self.shoreline_dir = "shoreline/validation"
        else:
            raise NotImplemented

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
            )

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        distributed: bool = False,
        local_rank: int = 0,
        snapshot_name: str = "",
    ) -> Dict:
        conf_name = os.path.splitext(os.path.basename(self.args.config))[0]
        val_dir = os.path.join(self.args.val_dir, conf_name, str(self.args.fold))
        os.makedirs(val_dir, exist_ok=True)
        dataset_dir = os.path.join(self.args.data_dir, self.dataset_dir)
        extra_context = model.module.extra_context
        # if self.args.local_rank == 0:
        #     csv_paths = glob.glob(os.path.join(val_dir, "*.csv"))
        #     for csv_file in csv_paths:
        #         os.remove(csv_file)
        if distributed:
            dist.barrier()
        for sample in tqdm(dataloader, position=0):
            scene_id = sample["name"][0]
            tgt_path = os.path.join(val_dir, f"{scene_id}.csv")
            # if os.path.exists(tgt_path) and datetime.datetime.fromtimestamp(os.path.getmtime(tgt_path) )> datetime.datetime.now() - datetime.timedelta(hours=10):
            #     continue
            mask_dict = predict_scene_and_return_mm(
                [model],
                scene_id=scene_id,
                dataset_dir=dataset_dir,
                use_fp16=self.args.fp16,
                rotate=True,
                crop_size=self.crop_size,
                overlap=self.overlap,
                extra_context=extra_context,
                iter_function=self.build_iterator,
                position=self.args.local_rank + 1,
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
        if distributed:
            dist.barrier()
        xview = 0
        output = {}
        if self.args.local_rank == 0:
            csv_paths = glob.glob(os.path.join(val_dir, "*.csv"))
            pred_csv = f"pred_{conf_name}_{self.args.fold}.csv"
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
                self.args.data_dir, self.annotation_dir
            )
            metric_args.shore_root = os.path.join(
                self.args.data_dir, self.shoreline_dir
            )
            metric_args.shore_tolerance = 2
            metric_args.costly_dist = True
            metric_args.drop_low_detect = True
            metric_args.distance_tolerance = 200
            metric_args.output = os.path.join(self.args.logdir, "out.json")
            output = xview_metric.evaluate_xview_metric(metric_args)
            xview = output["aggregate"]
        if distributed:
            dist.barrier()
        empty_cache()
        return {"xview": xview, **output}

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
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


def get_args_parser():
    parser = argparse.ArgumentParser("Pipeline", add_help=False)
    arg = parser.add_argument
    arg(
        "--config",
        metavar="CONFIG_FILE",
        help="path to configuration file",
        default="configs/vgg13.json",
    )
    arg("--workers", type=int, default=16, help="number of cpu threads to use")
    arg(
        "--gpu",
        type=str,
        default="0",
        help="List of GPUs for parallel training, e.g. 0,1,2,3",
    )
    arg("--output-dir", type=str, default="weights/")
    arg("--resume", type=str, default="")
    arg("--fold", type=int, default=0)
    arg("--crop_size_val", type=int, default=3584)
    arg("--overlap_val", type=int, default=704)
    arg("--prefix", type=str, default="val_")
    arg("--data-dir", type=str, default="/mnt/viper/xview3/")
    arg("--shoreline-dir", type=str, default="")
    arg("--val-dir", type=str, default="/mnt/viper/xview3/oof")
    arg("--folds-csv", type=str, default="folds4val.csv")
    arg("--logdir", type=str, default="logs")
    arg("--zero-score", action="store_true", default=False)
    arg("--from-zero", action="store_true", default=False)
    arg("--fp16", action="store_true", default=False)
    arg("--distributed", action="store_true", default=False)
    arg("--local_rank", default=0, type=int)
    arg("--world-size", default=1, type=int)
    arg("--test_every", type=int, default=1)
    arg("--freeze-epochs", type=int, default=0)
    arg("--multiplier", type=int, default=1)
    arg("--val", action="store_true", default=False)
    arg("--name", type=str, default="")
    arg("--freeze-bn", action="store_true", default=False)
    arg("--crop_size", type=int, default=1024)
    arg("--positive_ratio", type=float, default=0.5)
    arg("--epoch", type=int, default=None)
    arg("--bs", type=int, default=None)
    arg("--lr", type=float, default=None)
    arg("--eta_min", type=float, default=None)
    arg("--wd", dest="weight_decay", type=float, default=None)
    arg("--drop_path", type=float, default=None)
    arg("--pretrained", type=str, default="default")
    arg("--classifier_lr", type=float, default=None)
    arg("--warmup_epochs", type=int, default=None)
    arg("--test", action="store_true", default=False)

    return parser


def create_data_datasets(args):
    if args.shoreline_dir:
        print("Legacy Warning:shoreline_dir is no longer used")
    conf = load_config(args.config, args=args)
    if args.local_rank == 0:
        print("dataset config crop size", conf["crop_size"])
    if args.test:
        train_annotations = os.path.join(args.data_dir, "labels/public.csv")
        train_dataset = XviewValDataset(
            mode="train",
            dataset_dir=args.data_dir,
            fold=12345,
            folds_csv=args.folds_csv,
            annotation_csv=train_annotations,
            crop_size=conf["crop_size"],
            multiplier=conf["multiplier"],
        )
        val_dataset = TestDataset(os.path.join(args.data_dir, "images/public"))
    else:
        train_annotations = os.path.join(args.data_dir, "labels/validation.csv")
        train_dataset = XviewValDataset(
            mode="train",
            dataset_dir=args.data_dir,
            fold=args.fold,
            folds_csv=args.folds_csv,
            annotation_csv=train_annotations,
            crop_size=conf["crop_size"],
            multiplier=conf["multiplier"],
            positive_ratio=args.positive_ratio,
        )
        val_dataset = XviewValDataset(
            mode="val",
            dataset_dir=args.data_dir,
            fold=args.fold,
            folds_csv=args.folds_csv,
            annotation_csv=train_annotations,
            crop_size=conf["crop_size"],
        )
    return train_dataset, val_dataset


def make_folder(p):
    if not os.path.exists(p):
        os.mkdir(p)


def main(args):
    if args.local_rank == 0:
        make_folder(args.output_dir)
        make_folder(args.logdir)
    trainer_config = TrainConfiguration(
        config_path=args.config,
        crop_size=args.crop_size,
        gpu=args.gpu,
        resume_checkpoint=args.resume,
        prefix=args.prefix,
        world_size=args.world_size,
        test_every=args.test_every,
        local_rank=args.local_rank,
        distributed=args.distributed,
        freeze_epochs=args.freeze_epochs,
        log_dir=args.logdir,
        output_dir=args.output_dir,
        workers=args.workers,
        from_zero=args.from_zero,
        zero_score=args.zero_score,
        fp16=args.fp16,
        freeze_bn=args.freeze_bn,
        name=args.name if args.name else None,
    )
    data_train, data_val = create_data_datasets(args)
    seg_evaluator = XviewEvaluator(args)
    trainer = PytorchTrainer(
        train_config=trainer_config,
        evaluator=seg_evaluator,
        fold=args.fold,
        train_data=data_train,
        val_data=data_val,
        args=args,
    )
    if args.test:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data_val, shuffle=False
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
    if args.val:
        trainer.validate()
        return
    trainer.fit()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
