import os
import warnings
from pathlib import Path
from typing import List

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
import torch

import models
from inference.postprocessing import process_confidence
from inference.run_inference import predict_scene_and_return_mm
from training.config import load_config

warnings.filterwarnings("ignore")
import argparse

import pandas as pd

from training.utils import load_checkpoint


def process_scene(models: List[torch.nn.Module], args):
    test_dataset_dir = args.data_dir

    scene_id = args.scene_id
    mask_dict = predict_scene_and_return_mm(models, scene_id=scene_id, dataset_dir=test_dataset_dir, use_fp16=True,
                                            rotate=True)
    data = process_confidence(scene_id, None, mask_dict)
    df = pd.DataFrame(data, columns=["detect_scene_row", "detect_scene_column", "scene_id", "is_vessel", "is_fishing",
                                     "vessel_length_m", "confidence", "mean_obj", "mean_vessel","mean_fishing",
                                     "mean_length", "mean_center"])
    df = df.reset_index()
    print(data)
    df["is_vessel"] = (df.is_vessel) | (df.mean_vessel > 90)
    df["is_fishing"] = (df.is_fishing) | (df.mean_fishing > 80)
    df = df[["detect_scene_row", "detect_scene_column", "scene_id", "is_vessel", "is_fishing", "vessel_length_m", ]]
    Path(os.path.split(args.out_csv)[0]).mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)


def load_model(args, config_path, checkpoint):
    conf = load_config(config_path)
    model = models.__dict__[conf['network']](**conf["encoder_params"])
    model = model.cpu()
    load_checkpoint(model, checkpoint, strict=True, verbose=False)
    model = model.float().cuda()
    channels_last = conf["encoder_params"].get("channels_last", False)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model.eval()


def main():
    args = parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config_paths = [os.path.join("configs", f"{config}.json") for config in args.configs]
    checkpoint_paths = [os.path.join(args.weights_path, checkpoint) for checkpoint in args.checkpoints]
    models = [load_model(args, conf, checkpoint) for conf, checkpoint in zip(config_paths, checkpoint_paths)]
    process_scene(models, args)


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--configs', nargs='+')
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--checkpoints', nargs='+')
    arg('--weights-path', type=str, default="weights")
    arg('--scene_id', type=str)
    arg('--data-dir', type=str, default="/mnt/sota/datasets/xview3/test")
    arg('--out-csv', type=str, default="/mnt/sota/datasets/xview3/test_csvs/out.csv")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
