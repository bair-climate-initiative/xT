import re
import warnings
from typing import List

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import zoo
from inference.postprocessing import process_confidence
from inference.run_inference import predict_scene_and_return_mm
from training.config import load_config
import torch.distributed as dist

from training.utils import load_checkpoint

warnings.filterwarnings("ignore")
import argparse
import os

import pandas as pd
import torch.distributed


class TestDataset(Dataset):

    def __init__(self, root_dir):
        super().__init__()
        self.names = os.listdir(root_dir)

    def __getitem__(self, index):
        return self.names[index]

    def __len__(self):
        return len(self.names)


def process_distributed(models: List[torch.nn.Module], args):
    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    test_dataset_dir = args.data_dir
    test_dataset = TestDataset(test_dataset_dir)

    sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=1, sampler=sampler, shuffle=False, num_workers=1, pin_memory=False
    )

    for sample in tqdm(test_loader):
        scene_id = sample[0]
        mask_dict = predict_scene_and_return_mm(models, scene_id=scene_id, dataset_dir=test_dataset_dir, use_fp16=True,
                                                rotate=True)
        data = process_confidence(scene_id, None, mask_dict)
        pd.DataFrame(data, columns=["detect_scene_row", "detect_scene_column", "scene_id", "is_vessel", "is_fishing",
                                    "vessel_length_m", "confidence", "mean_obj", "mean_vessel", "mean_fishing",
                                  "mean_length", "mean_center"]).to_csv(os.path.join(out_dir, f"{scene_id}.csv"))


def load_model(args, config_path, checkpoint):
    conf = load_config(config_path)
    model = zoo.__dict__[conf['network']](**conf["encoder_params"])
    model = model.cuda()
    load_checkpoint(model, checkpoint)
    channels_last = conf["encoder_params"].get("channels_last", False)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                    find_unused_parameters=True)
    return model.eval()


def main():
    args = parse_args()
    init_gpu(args)
    config_paths = [os.path.join("configs", f"{config}.json") for config in args.configs]
    checkpoint_paths = [os.path.join(args.weights_path, checkpoint) for checkpoint in args.checkpoints]
    models = [load_model(args, conf, checkpoint) for conf, checkpoint in zip(config_paths, checkpoint_paths)]
    process_distributed(models, args)


def init_gpu(args):
    if args.distributed:
        dist.init_process_group(backend="nccl",
                                rank=args.local_rank,
                                world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def parse_args():
    parser = argparse.ArgumentParser("Pipeline")
    arg = parser.add_argument
    arg('--configs', nargs='+')
    arg('--workers', type=int, default=16, help='number of cpu threads to use')
    arg('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    arg('--checkpoints', nargs='+')
    arg('--weights-path', type=str, default="weights")
    arg('--fold', type=int, default=0)
    arg('--data-dir', type=str, default="/mnt/md0/datasets/xview3/test")
    arg('--out-dir', type=str, default="/mnt/md0/datasets/xview3/test_preds")
    arg('--fp16', action='store_true', default=False)
    arg('--distributed', action='store_true', default=False)
    arg("--local_rank", default=0, type=int)
    arg("--world-size", default=1, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
