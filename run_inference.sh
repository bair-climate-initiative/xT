#!/bin/bash
set -eo pipefail

# Execute xView3 inference pipeline using CLI arguments passed in
# 1) Path to directory with all data files for inference
# 2) Scene ID
# 3) Path to output CSV

if [ $# -lt 3 ]; then
    echo "run_inference.sh: [#1 Path to directory with all data files for inference] [#2 Scene ID] [#3 Path to output CSV]"
else

    python predict_test_single.py --data-dir "$1" --scene_id "$2" --out-csv "$3" \
    --checkpoints val_only_TimmUnet_nfnet_l0_99_xview val_only_TimmUnet_tf_efficientnetv2_l_in21k_99_last val_only_TimmUnet_tf_efficientnetv2_l_in21k_77_xview val_only_TimmUnet_tf_efficientnetv2_m_in21k_99_last val_only_TimmUnet_tf_efficientnet_b7_ns_77_xview \
    --configs nf0 v2ln v2ln v2m b7
fi