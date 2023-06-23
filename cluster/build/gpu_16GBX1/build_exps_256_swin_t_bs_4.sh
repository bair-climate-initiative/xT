CUDA_VISIBLE_DEVICES=0 WANDB_API_KEY=553070c5ef0d454bcb1e91afaabf2359ef69f4a0 singularity exec --nv --bind $WORKDIR:$WORKDIR \
        $HOME/detectron2.sif \
  python3 -m torch.distributed.launch\
        --nproc_per_node 1  \
        --master_port 10025 \
        train_val_segmentor.py  \
        --world-size 1   \
        --distributed  \
        --config configs/swin.json  \
        --workers 8  \
        --data-dir=$XVIEW_ROOT \
        --test_every 1 \
        --shoreline-dir $XVIEW_ROOT/shoreline/validation \
        --val-dir $HOME/output-swin_t_bs_4_ep_240_cp256-$ID  \
        --output-dir $HOME/output-swin_t_bs_4_ep_240_cp256-$ID \
        --folds-csv meta/folds.csv \
        --prefix val_only_  \
        --fold 77    \
        --freeze-epochs 0 \
        --fp16 --name swin_t_bs_4_ep_240_cp256 \
        --crop_size 256 \
        --crop_size_val 256 \
        --overlap_val 10 \
        --positive_ratio 0.8 \
        --test_every 20 \
        --epoch 240 \
        --bs 4 \
        --lr 0.003 \
        --wd 1.0e-4 \
        --drop_path 0.2 ;