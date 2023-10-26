#!/bin/bash
HOME=/home/tyler/xview3-detection
EXP_NAME=revswinv2_xl_4096_chip1024_swinpt_lr3e-3
DATA_DIR=/home/group/xview3

# * Modify args before --distributed for slurm-specific settings
# * Modify args after  --name for experiment-specific settings

# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
# WANDB_MODE=disabled \
python submitit_train_em.py \
    --job_dir $HOME/slurm/$EXP_NAME \
    --ngpus 2 \
    --nodelist em4 \
    --qos low \
    --debug False

# use the below for non-slurm launches.
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=6,7 \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port 47769 train_val_segmentor.py \
