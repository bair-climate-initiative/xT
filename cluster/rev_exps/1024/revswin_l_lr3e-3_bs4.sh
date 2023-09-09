NUM_GPUS=2
CONFIG=revswinv2_l
PORT=10025
NAME=revswin_l_lr3e-3_bs4
CROP=1024
CROP_VAL=1024
BS=4
EPOCH=60
LR=0.003
WD=1.0e-4
DROP_PATH=0.2
SAMPLE_RATE=0.5
REQUIRED_VRAM=32GB
PRETRAINED=/home/tyler/ckpts/swinv2_tiny_window16_256.pth
TEST_EVERY=10