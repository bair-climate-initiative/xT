./train_xview_single.sh  \
    1 \
    /home/group/xview3 \
    /home/group/xview3/shoreline/validation \
    output/revswin \
    77 \
    revswin \
    59999 \
    revswin-test \
    256 \
    --crop_size_val 256 --overlap_val 10
    # --crop_size_val 784 --overlap_val 10
    # <optional --resume ckpt name>
    # /shared/ritwik/data/xview3/shoreline/validation \