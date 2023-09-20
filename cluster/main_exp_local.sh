NAME="Example_Run"
SHELL_HOME=$HOME/xView3_second_place/cluster
. $SHELL_HOME/local_housekeeping.sh
. $SHELL_HOME/base_project_config.sh
. $1
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PORT=10085
. $SHELL_HOME/run.sh