#!/bin/bash
eval $MAIN_CMD  \
        --nproc_per_node $NUM_GPUS  \
        --master_port $PORT \
        train_val_segmentor.py  \
        --world-size $NUM_GPUS   \
        --distributed  \
        --config configs/${CONFIG}.json  \
        --workers 8  \
        --data-dir=$DATA_DIR  \
        --test_every 1 \
        --shoreline-dir $SHORE_DIR \
        --val-dir $VAL_OUT_DIR  \
        --output-dir $VAL_OUT_DIR \
        --folds-csv meta/folds.csv \
        --prefix val_only_  \
        --fold $FOLD    \
        --freeze-epochs 0 \
        --fp16 --name $NAME \
        --crop_size $CROP \
        --crop_size_val $CROP_VAL \
        --overlap_val 10 \
        --positive_ratio $SAMPLE_RATE \
        --test_every 20 \
        --epoch $EPOCH \
        --bs $BS \
        --lr $LR \
        --wd $WD \
        --drop_path $DROP_PATH \
        --pretrained $PRETRAINED;