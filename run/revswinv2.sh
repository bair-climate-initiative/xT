# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
WANDB_MODE=disabled \
./train_xview.sh  \
    8 \
    /home/group/xview3 \
    /home/group/xview3/shoreline/validation \
    output/revswinv2 \
    77 \
    revswinv2 \
    59999 \
    revswinv2-test \
    1280 \
    --crop_size_val 1024 --overlap_val 10 \
    --test_every 20 --epoch 240 \
    --bs 1 --lr 0.0003 --wd 1.0e-4
    # --crop_size_val 784 --overlap_val 10
    # <optional --resume ckpt name>
    # /shared/ritwik/data/xview3/shoreline/validation \