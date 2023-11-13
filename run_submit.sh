OUTPUT_DIR=outputs \
XVIEW3_ROOT=/home/group/xview3 \
INATURALIST_ROOT=/home/group/inaturalist2018 \
/home/tyler/miniconda3/envs/scalemae/bin/torchrun --nproc_per_node=8 --master_port 47769 \
  train_val_segmentor.py \
  "$@"
# WANDB_MODE=disabled \