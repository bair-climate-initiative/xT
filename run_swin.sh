WANDB_MODE=disabled \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
./train_xview.sh  \
    1 \
    /home/group/xview3 \
    /home/group/xview3/shoreline/validation \
    output/swin \
    77 \
    swin \
    56789 \
    swin-test \
    256 \
    # <optional --resume ckpt name>
    # /shared/ritwik/data/xview3/shoreline/validation \