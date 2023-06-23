CUDA_VISIBLE_DEVICES=$DEVICES WANDB_API_KEY=553070c5ef0d454bcb1e91afaabf2359ef69f4a0 singularity exec --nv --bind $WORKDIR:$WORKDIR \
        $HOME/detectron2.sif \
  python3 -m torch.distributed.launch\
        --nproc_per_node $NUM_GPUS  \
        --master_port $PORT \
        train_val_segmentor.py  \
        --world-size $NUM_GPUS   \
        --distributed  \
        --config configs/$CONFIG.json  \
        --workers 8  \
        --data-dir=$XVIEW_ROOT \
        --test_every 1 \
        --shoreline-dir $XVIEW_ROOT/shoreline/validation \
        --val-dir $HOME/output-$NAME-$ID  \
        --output-dir $HOME/output-$NAME-$ID \
        --folds-csv meta/folds.csv \
        --prefix val_only_  \
        --fold $FOLD    \
        --freeze-epochs 0 \
        --fp16 --name $NAME \
        --crop_size $CROP \
        --crop_size_val $CROP_VAL \
        --overlap_val 10 \
        --positive_ratio $SAMPLE_RATE \
        --test_every 20 \
        --epoch $EPOCH \
        --bs $BS \
        --lr $LR \
        --wd $WD \
        --drop_path $DROP_PATH \
        --pretrained $PRETRAINED ;