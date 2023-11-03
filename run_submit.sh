# WANDB_MODE=disabled \
/home/tyler/miniconda3/envs/scalemae/bin/torchrun --nproc_per_node=8 --master_port 47769 \
  train_val_segmentor.py \
  "$@"