import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src",
    default="/shared/ritwik/data/panda-prostate/train_mask_intersect.csv",
    type=str,
)
parser.add_argument("--ratio", default=0.9, type=float, help="Train ratio")

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(args.src)
    df = df[df.has_mask == True].copy()
    n = len(df)
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_cutoff = int(n * args.ratio)
    df_train = df.iloc[indices[:n_cutoff]]
    df_val = df.iloc[indices[n_cutoff:]]
    df_train.to_csv("panda_split_tain.csv", index=False)
    df_val.to_csv("panda_split_val.csv", index=False)
