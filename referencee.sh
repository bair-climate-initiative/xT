CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -u -m torch.distributed.launch  --nproc_per_node=4  --master_port 9987 \
 predict_test.py  --distributed \
 --world-size 4 \
 --data-dir /shared/ritwik/data/xview3/images/public \
 --out-dir  pred_1024 \
 --checkpoints /home/jacklishufan/xView3_second_place/weights/val_only_TimmUnet_tf_efficientnetv2_l_in21k_77_xview \
 --configs v2l

 python  \
 predict_test.py  --gpu 5 \
 --data-dir /shared/ritwik/data/xview3/images/public \
 --out-dir  pred_1024 \
 --checkpoints /home/jacklishufan/xView3_second_place/weights/val_only_TimmUnet_tf_efficientnetv2_l_in21k_77_xview \
 --configs v2l