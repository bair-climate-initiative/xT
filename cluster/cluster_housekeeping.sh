
ID=$(cat /proc/sys/kernel/random/uuid | sed 's/[-]//g' | head -c 5)
module load singularity

DATA_DIR=$WORKDIR/ritwik/data/xview3
SHORE_DIR=$WORKDIR/ritwik/data/xview3/shoreline/validation
VAL_OUT_DIR=$HOME/output-$RUN_NAME-$ID
# CD
cd $HOME/xview3_detection
# ENTRY
MAIN_CMD="WANDB_API_KEY=553070c5ef0d454bcb1e91afaabf2359ef69f4a0 singularity exec --nv --bind $WORKDIR:$WORKDIR \
    $HOME/detectron2.sif \
    python -u -m torch.distributed.launch"