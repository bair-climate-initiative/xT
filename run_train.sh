CUDA_LAUNCH_BLOCKING=1 WANDB_MODE=disabled \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
./train_xview.sh  \
    1 \
    /home/group/xview3 \
    /home/group/xview3/shoreline \
    output/revswin \
    77 \
    revswin \
    56789 \
    revswin-test \
    224 \
    # <optional --resume ckpt name>
    # /shared/ritwik/data/xview3/shoreline/validation \