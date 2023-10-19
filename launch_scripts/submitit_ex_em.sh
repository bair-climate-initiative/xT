#!/bin/bash
HOME=/home/tyler/xview3-detection
EXP_NAME=revswinv2_xl_4096_chip1024_swinpt_lr3e-3
DATA_DIR=/home/group/xview3

# * Modify args before --distributed for slurm-specific settings
# * Modify args after  --name for experiment-specific settings

# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
# WANDB_MODE=disabled \
python submitit_train.py \
    --job_dir $HOME/logs/$EXP_NAME \
    --ngpus 10 \
    --nodelist em4 \
    --qos low \
    --distributed \
    --workers 1 \
    --data-dir $DATA_DIR \
    --shoreline-dir $DATA_DIR/shoreline/validation \
    --val-dir output/$EXP_NAME \
    --output-dir $HOME/logs/$EXP_NAME \
    --folds-csv meta/folds.csv \
    --fold 77 \
    --name $EXP_NAME \
    --config configs/revswinv2_xl_4096.json \
    --crop_size 4096 \
    --bs 1 \
    --lr 0.003 \
    --wd 1e-4 \
    --test_every 1 \
    --epoch 10 \
    --positive_ratio 0.8 \
    --fp16

# use the below for non-slurm launches.
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=6,7 \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port 47769 train_val_segmentor.py \
