#!/bin/bash
#SBATCH --qos=cbmm
#SBATCH -p cbmm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
##SBATCH --constraint=high-capacity
#SBATCH --constraint=48GB
#SBATCH --mem=64G
#SBATCH --time=80:00:00
##SBATCH --exclude=node084,node092,node062,node066,node065,node077
#SBATCH --output=output/%j.sh

#SBATCH --mail-user=zss@mit.edu
#SBATCH --mail-type=ALL
date;hostname;id;pwd

echo 'activating virtual environment'
source ~/.bashrc
source activate pytorch-cuda-11.3

chmod u=rwx,g=r,o=r /om2/user/zss/ImplicitRankMinimization/program.sh
chmod u=rwx,g=r,o=r /om2/user/zss/ImplicitRankMinimization/train.py
module load openmind/gcc/11.1.0

echo 'running train.py'
echo "Saving to results/${xid}"

srun -n 1 python train.py --resume --results-path=results/${xid} > results/${xid}/print_${SLURM_PROCID}.txt 2>&1

