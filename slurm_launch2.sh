#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --qos low          # QOS (priority).
#SBATCH -N 1               # Number of nodes requested.
#SBATCH -n 4               # Number of tasks (i.e. processes).
#SBATCH --gres=gpu:4       # Number of GPUs.
##SBATCH --gpus-per-node=4
##SBATCH --cpus-per-task=4  # Number of cores per task.
##SBATCH --ntasks-per-node=4
## SBATCH -t 0-2:00          # Time requested (D-HH:MM).
##SBATCH --nodelist=em1,em2,em3,em9    # Uncomment if you need a specific machine.
#SBATCH --exclude=em8    # Uncomment if you need a specific machine.

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
# SBATCH -D /home/tyler/scale-mae-detection-xview

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.
#SBATCH -o slurm/%N_%j_%t.out    # STDOUT
#SBATCH -e slurm/%N_%j_%t.err    # STDERR

# Print some info for context.
pwd
hostname
date

echo "Starting job..."

source ~/.bashrc
conda activate scalemae2

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

# Do all the research.
# python train.py
bash run_submit.sh 4 $1 $2

# Print completion time.
date