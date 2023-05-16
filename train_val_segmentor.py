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
import torch.distributed as dist
from tqdm import tqdm

from inference.postprocessing import process_confidence
from inference.run_inference import predict_scene_and_return_mm
from metrics import xview_metric
from metrics.xview_metric import create_metric_arg_parser
from training.config import load_config
from training.val_dataset import XviewValDataset

warnings.filterwarnings("ignore")
import argparse
import os
from typing import Dict
import pandas as pd

from training.trainer import TrainConfiguration, PytorchTrainer, Evaluator

from torch.utils.data import DataLoader
import torch.distributed


class XviewEvaluator(Evaluator):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.crop_size = args.crop_size_val
        self.overlap = args.overlap_val

    def init_metrics(self) -> Dict:
        return {"xview": 0}

    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False, local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:
        conf_name = os.path.splitext(os.path.basename(self.args.config))[0]
        val_dir = os.path.join(self.args.val_dir, conf_name, str(self.args.fold))
        os.makedirs(val_dir, exist_ok=True)
        dataset_dir = os.path.join(self.args.data_dir, "images/validation")
        for sample in tqdm(dataloader):
            scene_id = sample["name"][0]
            mask_dict = predict_scene_and_return_mm([model], scene_id=scene_id, dataset_dir=dataset_dir,
                                                    use_fp16=self.args.fp16, rotate=True,
                                                    crop_size = self.crop_size,overlap=self.overlap)
            data = process_confidence(scene_id, None, mask_dict)
            pd.DataFrame(data,
                         columns=["detect_scene_row", "detect_scene_column", "scene_id", "is_vessel", "is_fishing",
                                  "vessel_length_m", "confidence", "mean_obj", "mean_vessel", "mean_fishing",
                                  "mean_length", "mean_center"]).to_csv(os.path.join(val_dir, f"{scene_id}.csv"))
        if distributed:
            dist.barrier()
        xview = 0
        output = {}
        if self.args.local_rank == 0:
            csv_paths = glob.glob(os.path.join(val_dir, "*.csv"))
            pred_csv = f"pred_{conf_name}_{self.args.fold}.csv"
            print(csv_paths)
            pd.concat([pd.read_csv(csv_path).reset_index() for csv_path in csv_paths]).to_csv(pred_csv, index=False)
            parser = create_metric_arg_parser()
            metric_args = parser.parse_args('')
            df = pd.read_csv(pred_csv)
            df = df.reset_index()
            df[["detect_scene_row", "detect_scene_column", "scene_id", "is_vessel", "is_fishing",
                 "vessel_length_m"]].to_csv(pred_csv, index=False)
            metric_args.inference_file = pred_csv
            metric_args.label_file = os.path.join(self.args.data_dir, "labels/validation.csv")
            metric_args.shore_root = self.args.shoreline_dir
            metric_args.shore_tolerance = 2
            metric_args.costly_dist = True
            metric_args.drop_low_detect = True
            metric_args.distance_tolerance = 200
            metric_args.output = "out.json"
            output = xview_metric.evaluate_xview_metric(metric_args)
            xview = output["aggregate"]
        if distributed:
            dist.barrier()
        empty_cache()
        return {"xview": xview,**output}

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        if current_metrics["xview"] > prev_metrics["xview"]:
            print("XView improved from {:.4f} to {:.4f}".format(prev_metrics["xview"], current_metrics["xview"]))
            improved["xview"] = current_metrics["xview"]
        else:
            print("XView {:.4f} current {:.4f}".format(prev_metrics["xview"], current_metrics["xview"]))
        return improved


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--config', metavar='CONFIG_FILE', help='path to configuration file', default="configs/vgg13.json")
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--output-dir', type=str, default='weights/')
    arg('--resume', type=str, default='')
    arg('--fold', type=int, default=0)
    arg('--crop_size_val', type=int, default=3584)
    arg('--overlap_val', type=int, default=704)
    arg('--prefix', type=str, default='val_')
    arg('--data-dir', type=str, default="/mnt/viper/xview3/")
    arg('--shoreline-dir', type=str, default="/mnt/viper/xview3/shore/validation")
    arg('--val-dir', type=str, default="/mnt/viper/xview3/oof")
    arg('--folds-csv', type=str, default='folds4val.csv')
    arg('--logdir', type=str, default='logs')
    arg('--zero-score', action='store_true', default=False)
    arg('--from-zero', action='store_true', default=False)
    arg('--fp16', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--world-size", default=1, type=int)
    arg("--test_every", type=int, default=1)
    arg('--freeze-epochs', type=int, default=0)
    arg('--multiplier', type=int, default=1)
    arg("--val", action='store_true', default=False)
    arg("--name", type=str, default='')
    arg("--freeze-bn", action='store_true', default=False)
    arg('--crop_size', type=int, default=1024)
    arg('--positive_ratio', type=float, default=0.5)
    

    args = parser.parse_args()

    return args


def create_data_datasets(args):
    conf = load_config(args.config)
    conf['crop_size'] = args.crop_size
    train_annotations = os.path.join(args.data_dir, "labels/validation.csv")
    train_dataset = XviewValDataset(mode="train", dataset_dir=args.data_dir, fold=args.fold, folds_csv=args.folds_csv,
                                    annotation_csv=train_annotations,
                                    crop_size=conf["crop_size"],
                                    multiplier=conf["multiplier"],
                                    positive_ratio=args.positive_ratio
                                    )
    val_dataset = XviewValDataset(mode="val", dataset_dir=args.data_dir, fold=args.fold, folds_csv=args.folds_csv,
                                  annotation_csv=train_annotations, crop_size=conf["crop_size"])
    return train_dataset, val_dataset


def make_folder(p):
    if not os.path.exists(p):
        os.mkdir(p)
def main():
    args = parse_args()
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
        name = args.name if args.name else None
    )

    data_train, data_val = create_data_datasets(args)
    seg_evaluator = XviewEvaluator(args)
    trainer = PytorchTrainer(train_config=trainer_config, evaluator=seg_evaluator, fold=args.fold,
                             train_data=data_train, val_data=data_val)
    if args.val:
        trainer.validate()
        return
    trainer.fit()


if __name__ == '__main__':
    main()
