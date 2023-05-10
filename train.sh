#!/bin/bash

NUM_GPUS=$1
DATA_DIR=$2
SHORE_DIR=$3
VAL_OUT_DIR=$4
FOLD=$5
CONFIG=$6

PYTHONPATH=.  python -u -m torch.distributed.launch  --nproc_per_node=$NUM_GPUS  --master_port 9989  train_val_segmentor.py  \
 --world-size $NUM_GPUS   --distributed  --config configs/${CONFIG}.json  --workers 8  --data-dir=$DATA_DIR  --test_every 1 \
--shoreline-dir $SHORE_DIR --val-dir $VAL_OUT_DIR --output_dir $VAL_OUT_DIR --folds-csv folds.csv --prefix val_only_  --fold $FOLD    --freeze-epochs 0 --name 256x256