#!/bin/bash
#SBATCH --job-name=example_experiment
#SBATCH --qos=frontier
#SBATCH --time=168:00:00
#SBATCH --account=ODEFN5169CYFZ

### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=mla
#SBATCH --chdir=/p/home/ritwik/dev/xview3-detection
#SBATCH --output=/p/home/ritwik/logs/example_experiment.log

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=56001
export NNODES=1
export GPUS_PER_NODE=4
export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed  
source /p/home/ritwik/miniconda3/etc/profile.d/conda.sh
conda activate scale

echo "Python version= " `python -V`

ID=$(cat /proc/sys/kernel/random/uuid | sed 's/[-]//g' | head -c 5)
NAME=example_experiment
NUM_GPUS=8
CONFIG=mae_hier
FOLD=77
NAME=mae_h_hpc
BS=4
EPOCH=800
LR=0.0003
WD=1.0e-4
PRETRAINED=false
TEST_EVERY=20
XVIEW_ROOT=/p/app/projects/nga-frontier/xview3

### the command to run
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    train_val_segmentor.py  \
        --distributed  \
        --config configs/$CONFIG.yaml  \
        --workers 1  \
        --data-dir=$XVIEW_ROOT \
        --test_every 1 \
        --shoreline-dir $XVIEW_ROOT/shoreline/validation \
        --val-dir $HOME/output-$NAME-$ID  \
        --output-dir $HOME/output-$NAME-$ID \
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
        --pretrained $PRETRAINED ${@:1};