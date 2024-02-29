torchrun --nproc_per_node=$1 --master_port $2 \
  train.py \
  "$@"
