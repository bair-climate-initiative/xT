#!/bin/bash
HOME=/p/home/ritwik
EXP_NAME=exp_name
PROJECT_DIR=/p/min-xT
PRETRAINED_CKPT_PATH=/p/path/to/pretrained_weights

CONSTRAINT=$1

### init virtual environment if needed  
source /p/path/to/conda.sh
conda activate xt

cd $PROJECT_DIR

# * Modify args before --distributed for slurm-specific settings
# * Modify args after  --name for experiment-specific settings

# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
NUMEXPR_MAX_THREADS=128 \
WANDB_MODE=offline \
PRETRAINED_CKPT_PATH=$PRETRAINED_CKPT_PATH \
EXP_NAME=$EXP_NAME \
PYTHONUNBUFFERED=1 \
python $PROJECT_DIR/launch_scripts/submitit_train_cluster.py \
    --job_dir /p/path/to/jobs/$EXP_NAME \
    --constraint $CONSTRAINT \
    --qos frontier \
    --account XYZ \
    --nodes 1 \
    --config $PROJECT_DIR/config/what/you/want.yaml
