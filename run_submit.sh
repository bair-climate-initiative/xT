OUTPUT_DIR=outputs \
WANDB_MODE=disabled \
torchrun --nproc_per_node=$1 --master_port $2 \
  train.py \
  "$@"

# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
