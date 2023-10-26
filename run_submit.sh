/home/tyler/miniconda3/envs/scalemae/bin/torchrun --nproc_per_node=1 --master_port 47769 \
  train_val_segmentor.py \
  config=config/base_config.yaml
#   name=revswinv2_chip512_swinpt_lr7e-3_repro \
#   optimizer.base_lr=7e-3 model/backbone=revswinv2_tiny_swinpt