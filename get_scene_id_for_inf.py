import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src", default="/shared/ritwik/data/xview3/labels/public.csv"
)

args = parser.parse_args()
df = pd.read_csv(args.src)
unique_scenes = np.unique(df.scene_id)

with open("unique_scene_xview.txt", "w") as f:
    f.write("\n".join(unique_scenes))
