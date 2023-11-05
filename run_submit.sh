OUTPUT_DIR=outputs \
XVIEW3_ROOT=/home/group/xview3 \
WANDB_MODE=disabled \
/home/tyler/miniconda3/envs/scalemae/bin/torchrun --nproc_per_node=2 --master_port 47769 \
  train_val_segmentor.py \
  "$@"