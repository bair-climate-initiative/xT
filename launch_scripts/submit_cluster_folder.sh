#!/bin/bash
HOME=/p/home/ritwik
PROJECT_DIR=/p/home/ritwik/dev/revswin-xl
PRETRAINED_CKPT_PATH=/p/home/ritwik/pretrained_weights
CONSTRAINT=$2

CMD_RUN_FROM=$(pwd)

### init virtual environment if needed  
source /p/home/ritwik/miniconda3/etc/profile.d/conda.sh
conda activate scale

for fpath in $(ls -1 $1)
do
    EXP_NAME=${fpath%.*}
    CONFIG_FILE_PATH=$(realpath $1)/$EXP_NAME.yaml
    echo ${CONFIG_FILE_PATH}

    cd $PROJECT_DIR

    NUMEXPR_MAX_THREADS=128 \
    WANDB_MODE=offline \
    PRETRAINED_CKPT_PATH=$PRETRAINED_CKPT_PATH \
    EXP_NAME=$EXP_NAME \
    python $PROJECT_DIR/launch_scripts/cluster/submitit_train_cluster.py \
        --job_dir /p/path/to/jobs/$EXP_NAME \
        --constraint $CONSTRAINT \
        --qos frontier \
        --account ODEFN5169CYFZ \
        --nodes 1 \
        --config $CONFIG_FILE_PATH

    cd $CMD_RUN_FROM
done
