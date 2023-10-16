# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
# WANDB_MODE=disabled \
./train_xview.sh  \
    10 \
    /home/group/xview3 \
    /home/group/xview3/shoreline/validation \
    output/revswinv2_xl_4096 \
    77 \
    revswinv2_xl_4096 \
    59999 \
    revswinv2_xl_4096_chip1024_swinpt_lr3e-3 \
    4096 \
    --crop_size_val 4096 --overlap_val 10 \
    --test_every 1 --epoch 10 \
    --bs 1 --lr 0.003 --wd 1.0e-4 \
    --positive_ratio 0.8 --fp16 
    # --classifier_lr 0.03 \
    # /shared/ritwik/data/xview3/shoreline/validation \