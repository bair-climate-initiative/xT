pbsdsh -v -- bash -l -c "module load singularity && CUDA_VISIBLE_DEVICES=0,1 singularity exec --nv --bind $WORKDIR:$WORKDIR $HOME/detectron2.sif \
	torchrun \
    --nnodes 2 \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    --node_rank=$PBS_NODENUM \
    --rdzv_id $PBS_JOBID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    train_val_segmentor.py  \
        --world-size 4   \
        --distributed  \
        --config configs/swin.json  \
        --workers 8  \
        --data-dir=$XVIEW_ROOT \
        --test_every 1 \
        --shoreline-dir $XVIEW_ROOT/shoreline/validation \
        --val-dir $HOME/output-exps_256_swin_t_bs_16-$ID  \
        --output-dir $HOME/output-exps_256_swin_t_bs_16-$ID \
        --folds-csv meta/folds.csv \
        --prefix val_only_  \
        --fold 77    \
        --freeze-epochs 0 \
        --fp16 --name exps_256_swin_t_bs_16 \
        --crop_size 256 \
        --crop_size_val 256 \
        --overlap_val 10 \
        --positive_ratio 0.8 \
        --test_every 20 \
        --epoch 240 \
        --bs 4 \
        --lr 0.003 \
        --wd 1.0e-4 \
        --drop_path 0.2 \
        --pretrained $SWINT_CKPT ;