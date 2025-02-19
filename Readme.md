# SGD and Weight Decay Secretly Compress Your Neural Network

## Requirements
- Python 3.10
- Pytorch 1.11
- Numpy
- Tqdm

## Running Experiments
There are three main ways to run experiments. The jobs and associated hyperparameters will be logged in `experiments.csv`.

1. **To submit the code as a job to slurm:**
    ```sh
    sh train.sh
    ```

2. **To submit a sweep of hyperparameters to queue to slurm:**
    ```sh
    python sweep.py
    ```

3. **To directly run the code (not recommended except for testing/debugging):**
    ```sh
    python train.py
    ```

## Other Commands

**To host a Jupyter notebook server:**
```sh
sbatch jupyter/jupyter.sh
```

To resubmit a failed experiment to slurm, resuming from the last checkpoint:
```sh
sbatch --export=ALL,xid=[XID] --job-name=[XID]_train retrain.sh
```

To find and replace xids within a Jupyter notebook:
```sh
sh jupyter/replace.sh < diffout.txt > results_plot.ipynb
```

Other Files

* jupyter/sweep_plot.ipynb: New Jupyter notebook that plots results of experiments specified by hyperparameters.
* jupyter/results_plot.ipynb: Jupyter notebook used to plot the results of experiments.
* conf/global_settings.py: A file that specifies the configuration parameters and hyperparameters.
* log_settings.py: Logs the current state of settings and saves to CSV.
* analysis_convergence.py: Contains functions that help in measuring the distance between the weights at epoch T and T+1.
* analysis_rank.py: Contains functions that help in measuring the ranks of the various matrices in the network.
* utils.py: Contains functions responsible for saving data, loading datasets, etc.
* models: Contains implementations of networks used in training.

## Reference
If you found this code useful, please cite the following paper:
```
@misc{galanti2023characterizingimplicitbiasregularized,
      title={SGD and Weight Decay Secretly Compress Your Neural Network}, 
      author={Tomer Galanti and Zachary S. Siegel and Aparna Gupte and Tomaso Poggio},
      year={2024},
      eprint={2206.05794},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2206.05794}, 
}
```
