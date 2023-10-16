ID=$(cat /proc/sys/kernel/random/uuid | sed 's/[-]//g' | head -c 5)
module load singularity
NAME=example_experiment
NUM_GPUS=1
CONFIG=swin

FOLD=77
PORT=10025
NAME=NO_NAME
BS=4
EPOCH=240
LR=0.003
WD=1.0e-4
PRETRAINED=default
TEST_EVERY=20

pbsdsh -v -- bash -l -c "module load singularity && CUDA_VISIBLE_DEVICES=$DEVICES singularity exec --nv --bind $WORKDIR:$WORKDIR $HOME/detectron2.sif \
	torchrun \
    --nnodes $NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    --node_rank=$PBS_NODENUM \
    --rdzv_id $PBS_JOBID \
    --rdzv_backend c10d \
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
        --pretrained $PRETRAINED ${@:2};