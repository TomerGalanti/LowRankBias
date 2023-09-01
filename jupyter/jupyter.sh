#!/bin/bash
#SBATCH --qos=cbmm
#SBATCH -p cbmm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=./jupyter/log_jupyter.sh
#SBATCH --exclude=node083

hostname

unset XDG_RUNTIME_DIR

source ~/.bashrc
source activate jupyter

jupyter lab --ip=0.0.0.0 --port=9000 --no-browser
