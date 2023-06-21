NAME="Example_Run"
SHELL_HOME=$HOME/xView3_second_place/cluster
. $SHELL_HOME/base_project_config.sh
. $SHELL_HOME/local_housekeeping.sh
. $1
export CUDA_VISIBLE_DEVICES=0,1,2,3
PORT=10057
. $SHELL_HOME/run.sh