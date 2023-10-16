# CUDA_LAUNCH_BLOCKING=1 \
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
    512 \
    --crop_size_val 512 --overlap_val 10 \
    --test_every 20 --epoch 240 \
    --bs 1 --lr 0.003 --wd 1.0e-4
    # --crop_size_val 784 --overlap_val 10
    # <optional --resume ckpt name>
    # /shared/ritwik/data/xview3/shoreline/validation \