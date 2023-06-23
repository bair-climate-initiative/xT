pbsdsh -v -- bash -l -c "module load singularity && CUDA_VISIBLE_DEVICES=0,1 singularity exec --nv --bind $WORKDIR:$WORKDIR $HOME/detectron2.sif \
	torchrun \
    --nnodes 4 \
    --nproc_per_node=2 \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
    --node_rank=$PBS_NODENUM \
    --rdzv_id $PBS_JOBID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:$MASTER_PORT" \
    train_val_segmentor.py  \
    --world-size 8   \
    --distributed  \
    --config configs/${CONFIG}.json  \
    --workers 8  \
    --data-dir=$DATA_DIR  \
    --test_every 1 \
    --shoreline-dir $SHORE_DIR \
    --val-dir $VAL_OUT_DIR  \
    --output-dir $VAL_OUT_DIR \
    --folds-csv meta/folds.csv \
    --prefix val_only_  \
    --fold 77    \
    --freeze-epochs 0 \
    --fp16 --name swin_t_bs_8_ep_240_cp256 \
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
    --pretrained default ;"