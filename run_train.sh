CUDA_LAUNCH_BLOCKING=1 \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
WANDB_MODE=disabled \
./train_xview.sh  \
    1 \
    /home/group/xview3 \
    /home/group/xview3/shoreline/validation \
    output/revswin \
    77 \
    revswin \
    59999 \
    revswin-test \
    224 \
    --crop_size_val 224 --overlap_val 10
    # --crop_size_val 784 --overlap_val 10
    # <optional --resume ckpt name>
    # /shared/ritwik/data/xview3/shoreline/validation \