WANDB_MODE=disabled \
./train_xview.sh  \
    8 \
    /home/group/xview3 \
    /home/group/xview3/shoreline/validation \
    output/swin \
    77 \
    swin_l_scratch \
    56789 \
    swin_l-test \
    256 \
    --crop_size_val 256 --overlap_val 10 \
    --test_every 20 --epoch 240 \
    --bs 5 --lr 0.003 --wd 1.0e-4
    # <optional --resume ckpt name>
    # /shared/ritwik/data/xview3/shoreline/validation \