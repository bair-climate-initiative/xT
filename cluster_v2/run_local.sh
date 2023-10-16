#!/bin/bash
MAIN_CMD=torchrun
NUM_GPUS="${NUM_GPUS:=8}"
PORT="${PORT:=10025}"
eval $MAIN_CMD  \
        --nproc_per_node $NUM_GPUS  \
        --master_port $PORT \
        train_val_segmentor.py  \
        --world-size $NUM_GPUS   \
        --distributed  \
        --config configs/${CONFIG}.yaml  \
        --workers 1  \
        --data-dir=$DATA_DIR  \
        --shoreline-dir $SHORE_DIR \
        --val-dir $VAL_OUT_DIR  \
        --output-dir $VAL_OUT_DIR \
        --folds-csv meta/folds.csv \
        --prefix val_only_  \
        --fold $FOLD    \
        --freeze-epochs 0 \
        --fp16 --name $NAME \
        --overlap_val 10 \
        --test_every $TEST_EVERY \
        --epoch $EPOCH \
        --bs $BS \
        --lr $LR \
        --wd $WD \
        --pretrained $PRETRAINED ${@:2};