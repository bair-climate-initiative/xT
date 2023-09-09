#PATH CONFIG
source ~/miniconda3/etc/profile.d/conda.sh
conda activate open-mmlab
DATA_DIR=/shared/group/xview3/
SHORE_DIR=/shared/group/xview3/shoreline/validation
VAL_OUT_DIR=/shared/jacklishufan/large-image-models/swin_t_512_p_09
# CD
cd $HOME/xView3_second_place
# ENTRY
MAIN_CMD="python -m torch.distributed.launch"
SWINT_CKPT=/shared/jacklishufan/large-image-models/swin_tiny_w16_256_pretrained.pth
SWINL_CKPT=/shared/jacklishufan/large-image-models/swin_large_w16_256_pretrained.pth
SWINv2T_CKPT=/home/tyler/ckpts/swinv2_tiny_window16_256.pth
SWINv2L_CKPT=/home/tyler/ckpts/swinv2_large_window16_256.pth