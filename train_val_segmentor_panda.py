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
from inference.run_inference import predict_scene_and_return_mm,predict_whole_image_panda
from metrics import xview_metric
from metrics.xview_metric import create_metric_arg_parser
from training.config import load_config
from training.panda import Panda

warnings.filterwarnings("ignore")
import argparse
import os
from typing import Dict
import pandas as pd

from training.trainer import TrainConfiguration, Evaluator
from training.trainer import PytorchTrainerPANDA as PytorchTrainer

from torch.utils.data import DataLoader
import torch.distributed
import torchmetrics
from training.utils import get_random_subset,all_gather
import numpy as np
from torchmetrics.functional.classification.jaccard import _jaccard_from_confmat

class PandaEvaluator(Evaluator):

    mean_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=6)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    cf_mat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=6)
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

    def init_metrics(self) -> Dict:
        return {"MIOU": 0,"ACC":0}

    def validate(self, dataloader: DataLoader, model: torch.nn.Module, distributed: bool = False, local_rank: int = 0,
                 snapshot_name: str = "") -> Dict:
        # conf_name = os.path.splitext(os.path.basename(self.args.config))[0]
        # val_dir = os.path.join(self.args.val_dir, conf_name, str(self.args.fold))
        # os.makedirs(val_dir, exist_ok=True)
        # dataset_dir = os.path.join(self.args.data_dir, "images/validation")
        cf_mat = torch.zeros(6,6)
        acc = 0
        tol = 0
        for sample in tqdm(dataloader):
            full_image = sample["image"][0]
            mask_dict = predict_whole_image_panda([model],full_image,
                                                    use_fp16=self.args.fp16, rotate=False)
            mask_src = mask_dict['mask']
            mask_tgt = sample['mask'][0].cpu()
            cf_mat += self.cf_mat(torch.tensor(mask_src),mask_tgt)
            pred_score = Panda.isup_grade_from_mask(mask_src)
            target_score = sample['isup_grade'][0].item()
            if pred_score == target_score:
                acc += 1
            tol += 1
        
        cf_mat = all_gather(cf_mat)
        cf_mat = torch.stack(cf_mat).sum(0)
        print(cf_mat)
        acc = sum(all_gather(acc))
        tol = sum(all_gather(tol))
        miou = _jaccard_from_confmat(cf_mat,cf_mat.shape[0]).item()
        acc = acc / (tol + 1e-9)
        return {"MIOU": miou,"ACC": acc }

    def get_improved_metrics(self, prev_metrics: Dict, current_metrics: Dict) -> Dict:
        improved = {}
        for k in ["MIOU","ACC"]:
            if current_metrics[k] > prev_metrics[k]:
                print("{} improved from {:.4f} to {:.4f}".format(k,prev_metrics[k], current_metrics[k]))
                improved[k] = current_metrics[k]
            else:
                print("{} {:.4f} current {:.4f}".format(k,prev_metrics[k], current_metrics[k]))
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
    arg('--prefix', type=str, default='val_')
    arg('--data-dir', type=str, default="shared/ritwik/data/panda-prostate/")
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
    arg('--eval_size', type=int, default=10)

    args = parser.parse_args()

    return args


def create_data_datasets(args):
    conf = load_config(args.config)
    conf['crop_size'] = args.crop_size
    #train_annotations = os.path.join(args.data_dir, "train_mask_intersect.csv")
    train_annotations = 'panda_split_tain.csv'
    test_annotations = 'panda_split_val.csv'
    train_dataset = Panda(root_dir=args.data_dir,split='train',split_file=train_annotations,
                                    mode = 'random',
                                    crop_size=conf["crop_size"],
                                    )
    val_dataset = Panda(root_dir=args.data_dir,split='train',split_file=test_annotations,
                                    mode = 'full',
                                    crop_size=conf["crop_size"],
                                    )
    if args.eval_size:
        val_dataset = get_random_subset(val_dataset,args.eval_size)
    return train_dataset, val_dataset


def main():
    args = parse_args()
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
    seg_evaluator = PandaEvaluator(args)
    trainer = PytorchTrainer(train_config=trainer_config, evaluator=seg_evaluator,
                             train_data=data_train, val_data=data_val)
    if args.val:
        trainer.validate()
        return
    trainer.fit()


if __name__ == '__main__':
    main()
