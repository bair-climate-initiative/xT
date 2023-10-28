#!/bin/bash
HOME=/p/home/ritwik
EXP_NAME=revswinv2_xl_4096_chip1024_swinpt_lr3e-3
DATA_DIR=/p/app/projects/nga-frontier/xview3
PROJECT_DIR=/p/home/ritwik/dev/xview3-detection

CONSTRAINT=$1

### init virtual environment if needed  
source /p/home/ritwik/miniconda3/etc/profile.d/conda.sh
conda activate scale

cd $PROJECT_DIR

# * Modify args before --distributed for slurm-specific settings
# * Modify args after  --name for experiment-specific settings

# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
NUMEXPR_MAX_THREADS=128 \
WANDB_MODE=offline \
EXP_NAME=$EXP_NAME python $PROJECT_DIR/launch_scripts/cluster/submitit_train_cluster.py \
    --job_dir $HOME/logs/$EXP_NAME \
    --constraint $CONSTRAINT \
    --qos frontier \
    --account ODEFN5169CYFZ \
    --nodes 1 \
    --config $PROJECT_DIR/config/revswin_xl/4096_chip512_lr1e-3.yaml

# use the below for non-slurm launches.
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=6,7 \
# python -m torch.distributed.launch --nproc_per_node=8 --master_port 47769 train_val_segmentor.py \