import glob
import os

import pandas as pd
from torch.utils.data import Dataset

from inference.postprocessing import process_confidence
from inference.run_inference import predict_scene_and_return_mm
from metrics import xview_metric
from metrics.xview_metric import create_metric_arg_parser
from training.config import load_config
from training.val_dataset import XviewValDataset

val_dir = "pred_1024"
csv_paths = glob.glob(os.path.join(val_dir, "*.csv"))
pred_csv = f"pred_1024_temp.csv"
data_dir = "/shared/ritwik/data/xview3"
shoreline_dir = "/shared/ritwik/data/xview3/shoreline/public"

print(csv_paths)
pd.concat([pd.read_csv(csv_path).reset_index() for csv_path in csv_paths]).to_csv(
    pred_csv, index=False
)
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
metric_args.label_file = os.path.join(data_dir, "labels/public.csv")
metric_args.shore_root = shoreline_dir
metric_args.shore_tolerance = 2
metric_args.costly_dist = True
metric_args.drop_low_detect = True
metric_args.distance_tolerance = 200
metric_args.output = "out.json"
output = xview_metric.evaluate_xview_metric(metric_args)
xview = output["aggregate"]
