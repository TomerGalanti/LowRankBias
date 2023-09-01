""" configurations for this project
author baiyu
"""
import os
from datetime import datetime
import math

dataset_name = 'CIFAR10' # MNIST, FashionMNIST, STL10, CIFAR10, CIFAR100, SVHN

if dataset_name == 'MNIST' or dataset_name == 'FashionMNIST':
    num_output_classes = 10
    num_input_channels = 1
    mean = 0.1307
    std = 0.3081

elif dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100':
    num_output_classes = int(dataset_name[5:])
    num_input_channels = 3
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

elif dataset_name == 'STL10':
    num_output_classes = 10
    num_input_channels = 3
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

elif dataset_name == 'SVHN':
    num_output_classes = 10
    num_input_channels = 3
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

# device
device = 'cuda'

# runs
EPOCH = 500
SAVE_EPOCH = 10
shuffle = True

# early stop at EARLY_STOP_EPOCH or beyond if train acc falls below EARLY_STOP_ACC for the past 3 checkpoints.
EARLY_STOP_EPOCH = 100
EARLY_STOP_ACC = 0.15

# hyperparameters
MILESTONES = [60, 100, 200]
batch_size = 4
test_batch_size = 128
lr = 0.8
momentum = 0
warm = 1
tolerance = 0.001
weight_decay = 5e-4
rnd_aug = False
loss = 'CE' # 'CE', 'MSE'

# parameters for computing theoretical bound
bound_batch_size = 128
bound_num_batches = 100
rank_bound_version = "D0"

# architecture
net = 'resnet'

# mlp, convnet, convnet_custom parameters
depth = 10
width = 100
alpha = 1
activation = 'leaky_relu'
bias = True
bn = True

# mlp_custom, convnet_custom parameters
custom_layer = 5
custom_width = 100

# convnet_custom parameters
kernel_dim = 3
stride_length = 1
padding = 1

# resnet parameters
resnet_version = 18

# saving params
trial = 4
directory = './results/'
resume = False
normalize_dist = True
