#!/bin/bash
#SBATCH --qos=cbmm
##SBATCH -p cbmm
#SBATCH -p normal
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
##SBATCH --constraint=48GB
#SBATCH --constraint=high-capacity
#SBATCH --time=48:00:00
#SBATCH --exclude=node078
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
echo "Saving to ${dir_path}"

# if the job id file doesn't exist, start from scratch. otherwise, resume training.
FILE=${dir_path}/${SLURM_JOB_ID}.jobid
if test -f "$FILE"; then
  echo "Resuming from checkpoint"
  srun -n 1 python train.py --resume --results-path=${dir_path} > ${dir_path}/print_${SLURM_PROCID}.txt 2>&1
else
  echo "Training new model"
  touch $FILE
  srun -n 1 python train.py --results-path=${dir_path} > ${dir_path}/print_${SLURM_PROCID}.txt 2>&1
fi

