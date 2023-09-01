Requirements:
-- Python 3.10
-- Pytorch 1.11
-- Numpy
-- Tqdm

=== Running Experiments ===
There are three main ways to run experiments. The jobs and associated hyperparams will be logged in 'experiments.csv'

1) To submit the code as a job to slurm:
sh train.sh

2) To submit a sweep of hyperparameters to queue to slurm:
python sweep.py

3) To directly run the code (not recommended except for testing/debugging):
python train.py

=== Other Commands  ===

To host a Jupyter notebook server:
sbatch jupyter/jupyter.sh

To resubmit a failed experiment to slurm, resuming from last checkpoint:
sbatch --export=ALL,xid=[XID] --job-name=[XID]_train retrain.sh

To find and replace xids within a Jupyter notebook:
sh jupyter/replace.sh < diffout.txt > results_plot.ipynb

=== Other Files ===

jupyter/sweep_plot.ipynb:
New Jupyter notebook that plots results of experiments specified by hyperparams

jupyter/results_plot.ipynb:
Jupyter notebook used to plot the results of experiments

conf/global_settings.py:
A file that specifies the configuration parameters
and hyperparameters. 

log_settings.py:
Logs the current state of settings and saves to csv.

analysis_convergence.py:
Contains functions that help in measuring
the distance between the weights at epoch T and T+1.

analysis_rank.py:
Contains functions that help in measuring
the ranks of the various matrices in the network.

utils.py:
Contains functions responsible for saving data,
loading datasets, etc'.

models:
Contains implementations of networks used in training.


