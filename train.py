import time
from shutil import copyfile
import os

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import importlib
from tqdm import tqdm

import utils
from utils import get_network, get_training_dataloader, \
    get_test_dataloader, save_data
import analysis_rank, analysis_convergence
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Graphs:
    def __init__(self):
        self.train_accuracy = []
        self.test_accuracy = []

        self.train_loss = []
        self.test_loss = []

        self.rank = []
        self.avg_rank = []
        self.f_norm = []

        self.stable_rank = []
        self.avg_stable_rank = []

        self.singvals = []

        self.complexity = []

        self.dist = []

        self.num_matrices = 0

    def update_num_matrices(self, num_matrices):
        self.num_matrices = num_matrices
        self.rank = [[] for _ in range(num_matrices)]
        self.f_norm = [[] for _ in range(num_matrices)]
        self.stable_rank = [[] for _ in range(num_matrices)]
        self.singvals = [[] for _ in range(num_matrices)]

    def add_data(self, train_acc, test_acc,
                 train_loss, test_loss,
                 rank, avg_rank, stable_rank, 
                 avg_stable_rank, dist, 
                 singvals, complexity,
                 f_norm):

        if train_acc != None: self.train_accuracy += [train_acc]
        if test_acc != None: self.test_accuracy += [test_acc]
        if train_loss != None: self.train_loss += [train_loss]
        if test_loss != None: self.test_loss += [test_loss]
        if avg_rank != None: self.avg_rank += [avg_rank]
        if avg_stable_rank != None: self.avg_stable_rank += [avg_stable_rank]
        if complexity != None: self.complexity += [complexity]
        if dist != None: self.dist += [dist]

        for i in range(self.num_matrices):
            if rank != None: self.rank[i] += [rank[i]]
            if f_norm != None: self.f_norm[i] += [f_norm[i]]
            if stable_rank != None: self.stable_rank[i] += [stable_rank[i]]
            if singvals != None: self.singvals[i] += [singvals[i]]


def train_epoch(epoch, net, optimizer, train_loader, settings):
    start = time.time()
    if settings.loss == 'CE': loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE': loss_function = nn.MSELoss()
    net.train()
    for batch_index, (images, labels) in enumerate(tqdm(train_loader, leave=False, desc="  ")):

        if settings.device == 'cuda':
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)

        if settings.loss == 'CE':
            loss = loss_function(outputs, labels)
        elif settings.loss == 'MSE':
            loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

        loss.backward()
        optimizer.step()

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s\tLoss: {:0.4f}'.format(epoch, finish - start, loss.item()))


@torch.no_grad()
def eval_training(epoch, net, test_loader, settings):

    start = time.time()
    if settings.loss == 'CE':
        loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE':
        loss_function = nn.MSELoss()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in test_loader:

        if settings.device == 'cuda':
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        if settings.loss == 'CE':
            loss = loss_function(outputs, labels)
        elif settings.loss == 'MSE':
            loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()

    dataset_size = len(test_loader.dataset)
    acc = correct / dataset_size
    test_loss = test_loss / dataset_size

    finish = time.time()
    if settings.device == 'cuda':
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss,
        acc,
        finish - start
    ))
    print()

    return acc, test_loss

@torch.no_grad()
def save_checkpoint(path, net, graphs, optimizer, epoch):
    path += '/model.pt'
    torch.save({'model_state_dict': net.state_dict(),
                'graphs': graphs,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch}, path)

def main(results_path, resume):

    ## get directory name
    if results_path is None:
        directory = './results'
        dir_name = utils.get_dir_name(directory, prespecified=False)

        copyfile('./conf/global_settings.py', dir_name + '/global_settings.py')
    else:
        dir_name = utils.get_dir_name(results_path, prespecified=True, resume=resume)

        copyfile(os.path.join(results_path,'global_settings.py'), dir_name + '/global_settings.py')

    spec = importlib.util.spec_from_file_location("module", dir_name + '/global_settings.py')
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    
    net = get_network(settings)
    graphs = Graphs()

    # data preprocessing:
    train_loader = get_training_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=settings.batch_size,
        shuffle=settings.shuffle,
        rnd_aug=settings.rnd_aug,
        num_classes=settings.num_output_classes
    )

    test_loader = get_test_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=settings.test_batch_size,
        shuffle=True,
        num_classes=settings.num_output_classes
    )

    num_matrices = net.num_matrices
    graphs.update_num_matrices(num_matrices)

    optimizer = optim.SGD(net.parameters(), lr=settings.lr, momentum=settings.momentum,
                          weight_decay=settings.weight_decay)

    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)  # learning rate decay

    dist = 0
    start_epoch = 1
    skip_save = False

    # start from checkpoint
    if resume:
        checkpoint = torch.load(dir_name + '/model.pt')
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        graphs = checkpoint['graphs']
        start_epoch = checkpoint['epoch']
        skip_save = True # prevents the results from getting double-saved when resuming from checkpoint


    for epoch in tqdm(range(start_epoch, settings.EPOCH + 1)):
        train_scheduler.step(epoch)

        if epoch % settings.SAVE_EPOCH == 1 and not skip_save:

            rank = analysis_rank.eval_ranks(net, tol=settings.tolerance)
            f_norm = analysis_rank.eval_f_norm(net, settings.net)
            singvals = analysis_rank.eval_singvals(net)
            stable_rank = analysis_rank.eval_stable_ranks(singvals)

            avg_rank = np.mean(rank)
            avg_stable_rank = np.mean(stable_rank)

            complexity = analysis_rank.eval_complexity(net, settings.net) 

            train_acc, train_loss = eval_training(epoch, net, train_loader, settings)
            test_acc, test_loss = eval_training(epoch, net, test_loader, settings)

            graphs.add_data(train_acc, test_acc,
                            train_loss, test_loss,
                            rank, avg_rank, 
                            stable_rank, avg_stable_rank, 
                            dist, singvals, 
                            complexity, f_norm)

            save_data(dir_name, graphs)
            save_checkpoint(dir_name, net, graphs, optimizer, epoch)

        # evaluate dist and convergance
        if ('normalize_dist' in vars(settings)):
            normalize_dist = settings.normalize_dist
        else:
            normalize_dist = False
        list_weights_old = analysis_convergence.get_weights(net)
        train_epoch(epoch, net, optimizer, train_loader, settings)
        list_weights_new = analysis_convergence.get_weights(net)
        dist = analysis_convergence.eval_dist(list_weights_old, list_weights_new, normalize_dist)

        skip_save = False

        # early stop
        if ('EARLY_STOP_EPOCH' in vars(settings)):
            if epoch >= settings.EARLY_STOP_EPOCH and graphs.train_accuracy[-1] <= settings.EARLY_STOP_ACC and graphs.train_accuracy[-2] <= settings.EARLY_STOP_ACC and graphs.train_accuracy[-3] <= settings.EARLY_STOP_ACC:
                print("=== Initiating Early Stop ===")
                _ = open(dir_name + '/EARLY_STOP', 'w')
                _.close()
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-path', action='store', default=None,
                        help='Specifies path of directory where results should be stored (including weights) and '
                             'where the global_settings.py file is located. Default behavior is to use '
                             'conf/global_settings.py and create a new directory in results.')
    parser.add_argument('--resume', action=argparse.BooleanOptionalAction, default=False)
    _args = parser.parse_args()

    if _args.resume and _args.results_path is None:
        raise Exception("Cannot resume training without specifying results path.")

    main(_args.results_path, _args.resume)
