NAME="Example_Run"
SHELL_HOME=$HOME/xview3_detection/cluster
. $SHELL_HOME/base_project_config.sh
NAME="Example_Run"
. $SHELL_HOME/cluster_housekeeping.sh
# manually pass args here or use a loop instead
. $1
export CUDA_VISIBLE_DEVICES=6
PORT=10056
. $SHELL_HOME/run.sh