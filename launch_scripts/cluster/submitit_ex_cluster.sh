#!/bin/bash
HOME=/p/home/ritwik
EXP_NAME=revswinv2_xl_4096_chip1024_swinpt_lr3e-3
DATA_DIR=/p/app/projects/nga-frontier/xview3
PROJECT_DIR=/p/home/ritwik/dev/xview3-detection

CONSTRAINT=$1

### init virtual environment if needed  
source /p/home/ritwik/miniconda3/etc/profile.d/conda.sh
conda activate scale

# * Modify args before --distributed for slurm-specific settings
# * Modify args after  --name for experiment-specific settings

# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
NUMEXPR_MAX_THREADS=128 \
WANDB_MODE=offline \
python submitit_train_cluster.py \
    --job_dir $HOME/logs/$EXP_NAME \
    --constraint viz \
    --qos frontier \
    --account ODEFN5169CYFZ \
    --nodes 1 \
    --distributed \
    --workers 1 \
    --data-dir $DATA_DIR \
    --shoreline-dir $DATA_DIR/shoreline/validation \
    --output-dir $HOME/logs/$EXP_NAME \
    --log-dir $HOME/logs/$EXP_NAME \
    --folds-csv $PROJECT_DIR/meta/folds.csv \
    --fold 77 \
    --name $EXP_NAME \
    --config $PROJECT_DIR/configs/revswinv2_xl_4096.json \
    --crop_size 4096 \
    --bs 4 \
    --lr 0.003 \
    --wd 1e-4 \
    --test_every 1 \
    --epoch 10 \
    --fp16

# use the below for non-slurm launches.
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=6,7 \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port 47769 train_val_segmentor.py \
