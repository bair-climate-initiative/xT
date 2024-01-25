OUTPUT_DIR=outputs \
XVIEW3_ROOT=/datasets/xview3_2024-01-10_1001/ \
/home/tyler/miniconda3/envs/scalemae2/bin/torchrun --nproc_per_node=$1 --master_port $2 \
  train_val_segmentor.py \
  "$@"

# CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
