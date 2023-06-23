#!/bin/bash

NUM_GPUS=$1
DATA_DIR=$2
SHORE_DIR=$3
VAL_OUT_DIR=$4
FOLD=$5
CONFIG=$6
PORT=$7
NAME=$8
CROP=$9

# missing --distributed
PYTHONPATH=.  python -u -m torch.distributed.launch  --nproc_per_node=$NUM_GPUS  --master_port $PORT  train_val_segmentor.py  \
 --world-size $NUM_GPUS --config configs/${CONFIG}.json  --workers 8  --data-dir=$DATA_DIR  --test_every 1 \
--shoreline-dir $SHORE_DIR --val-dir $VAL_OUT_DIR  --output-dir $VAL_OUT_DIR --folds-csv meta/folds.csv --prefix val_only_  --fold $FOLD  --freeze-epochs 0 --fp16 --name $NAME \
--crop_size $CROP ${@:10}