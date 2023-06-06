NAME="Example_Run"
SHELL_HOME=$HOME/xView3_second_place/cluster
. $SHELL_HOME/base_project_config.sh
. $SHELL_HOME/local_housekeeping.sh
. $1
export CUDA_VISIBLE_DEVICES=6
PORT=10056
. $SHELL_HOME/run.sh