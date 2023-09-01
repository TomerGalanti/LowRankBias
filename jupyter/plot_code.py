import numpy as np
import ast
import matplotlib
import matplotlib.pyplot as plt
import os
import os.path
import matplotlib.ticker as ticker

num_checkpoints = 50
save_skip = 10
epochs_full = [i*save_skip for i in range(1,num_checkpoints+1)]
cmap = plt.get_cmap("tab10")
cmap2 = plt.get_cmap("tab20")
results_directory = "../results/"

# Verifies whether the given experiment is finished training, prints a warning if not
def check_training_progress(xid, pid):
    vals = load_data(xid, pid, 'avg_rank')
    
    if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))):
        print("Warning: Experiment " + str(xid) + "/" + str(pid) + " does not " + 
              "exist or training has not commenced.")
        return
    if len(vals) < num_checkpoints and not os.path.isfile(os.path.join(results_directory, str(xid), str(pid), 'EARLY_STOP')):
        print("Warning: Experiment " + str(xid) + "/" + str(pid) + 
              " has not finished training. (Epoch " + str((len(vals)-1)*save_skip) + 
              " / " + str(num_checkpoints*save_skip) + ")")
    if len(vals) > num_checkpoints:
        print("Warning: Experiment " + str(xid) + "/" + str(pid) + 
              " has duplicate entries.")
    if os.path.isfile(os.path.join(results_directory, str(xid), str(pid), 'EARLY_STOP')):
        return "early_stop"


# Loads the data from .txt file in results directory
def load_data(xid, pid, plot_name):
    filepath = os.path.join(results_directory, str(xid), str(pid), plot_name + ".txt")
    if not os.path.isfile(filepath): return []
    
    f = open(filepath, "r")
    st = f.read().replace('nan', '0')
    vals = np.array(ast.literal_eval(st))
    
    return vals


# Returns the number of layers in the network
def num_layers(xid, pid):
    if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))): return 0
    
    num_layers = 0
    for filename in os.listdir(os.path.join(results_directory, str(xid), str(pid))):
        if "singvals_" in filename: num_layers+=1
    
    return num_layers


# Returns a 2D array of singular values, indexed by layer
def load_singvals(xid, pid):    
    singvals = []
    for layer in range(num_layers(xid, pid)):
        singvals.append(load_data(xid, pid, "singvals_"+str(layer)))
    
    return singvals


# Computes the rank from the singular values by layer
def compute_rank(xid, pid, tol):
    singvals = load_singvals(xid, pid)
    rank_by_layer = []
    
    for layer in range(len(singvals)):
        rank_by_layer.append([])
        layer_singvals = singvals[layer]
        
        for epoch in layer_singvals:
            rank = 0
            for singval in epoch:
                if singval > tol:
                    rank+=1
            rank_by_layer[layer].append(rank)
    
    return np.array(rank_by_layer)


# Computes the average rank across all layers
def compute_avg_rank(xid, pid, tol):
    rank_by_layer = compute_rank(xid, pid, tol)
    
    rank = [0]*len(rank_by_layer[0])
    for layer in rank_by_layer:
        element_num = 0
        for element in layer:
            rank[element_num] += element
            element_num += 1
    
    for layer in range(len(rank)):
        rank[layer] /= num_layers(xid, pid)
        
    return np.array(rank)


# Plot accuracy
def plot_accuracy(plot_name, parameter_name, parameters, xids, pid, save_directory, filetype):
    if len(parameters) != len(xids): raise ValueError("Parameters and xids must be of same length")
    
    for xid in xids: check_training_progress(xid, pid)
    
    for plot_type in ['train_accuracy', 'test_accuracy']:
        for i, xid in enumerate(xids):
            if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))): continue
            
            accuracy = load_data(xid, pid, plot_type)
            if plot_type == 'train_accuracy':
                style = '-'
                label = 'train ' + parameter_name + ' ' + str(parameters[i])
            elif plot_type == 'test_accuracy':
                style = '--'
                label = 'test ' + parameter_name + ' ' + str(parameters[i])
            
            accuracy = accuracy[:num_checkpoints] # truncate in case length exceeds max
            epochs = epochs_full[:len(accuracy)] # truncate in case not finished training
            plt.errorbar(epochs, accuracy, label=label, color=cmap(i), linestyle=style)
            
            plt.xlim(1,num_checkpoints*save_skip)
            plt.ylim(0, 1.05)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Accuracy', fontsize=15)
            plt.legend(loc='upper right', prop={'size':14})
            
    filename = "acc_" + plot_name + "_" + parameter_name.replace(" ", "_") + "." + filetype
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, filename))
    plt.show()


# Plot loss
def plot_loss(plot_name, parameter_name, parameters, xids, pid, save_directory, filetype):
    if len(parameters) != len(xids): raise ValueError("Parameters and xids must be of same length")
    
    for xid in xids: check_training_progress(xid, pid)
    
    for plot_type in ['train_loss', 'test_loss']:
        for i, xid in enumerate(xids):
            if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))): continue
            
            loss = load_data(xid, pid, plot_type)
            if plot_type == 'train_loss':
                style = '-'
                label = 'train; ' + parameter_name + ' ' + str(parameters[i])
            elif plot_type == 'test_loss':
                style = '--'
                label = 'test; ' + parameter_name + ' ' + str(parameters[i])
            
            loss = loss[:num_checkpoints] # truncate in case length exceeds max
            epochs = epochs_full[:len(loss)] # truncate in case not finished training
            plt.errorbar(epochs, loss, label=label, color=cmap(i), linestyle=style)
            
            plt.xlim(1,num_checkpoints*save_skip)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Loss', fontsize=15)
            plt.legend(loc='upper right', prop={'size': 6})
            
    filename = "loss_" + plot_name + "_" + parameter_name.replace(" ", "_") + "." + filetype
    plt.savefig(os.path.join(save_directory, filename))
    plt.show()

# Plot complexity
def plot_complexity(plot_name, parameter_name, parameters, xids, pid, save_directory, filetype, tol=0.001, plot_rank=False):
    if len(parameters) != len(xids): raise ValueError("Parameters and xids must be of same length")
    
    for xid in xids: check_training_progress(xid, pid)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    if plot_rank: ax2 = ax1.twinx()

    for i, xid in enumerate(xids):
        if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))): continue
        
        # Plot the Complexity
        complexity = load_data(xid, pid, "complexity")
        
        style = '-'
        label = 'complexity; ' + parameter_name + ' ' + str(parameters[i])
        
        complexity = complexity[:num_checkpoints] # truncate in case length exceeds max
        epochs = epochs_full[:len(complexity)] # truncate in case not finished training
        
        l1 = ax1.errorbar(epochs, complexity, label=label, color=cmap(i), linestyle=style)
        ax1.set_ylabel("Complexity", fontsize=15)
        ax1.semilogy()
        
        # Plot the Rank
        if plot_rank:
            ranks = compute_avg_rank(xid, pid, tol)
            style = '--'
            label = 'rank; ' + parameter_name + ' ' + str(parameters[i])
            ranks = ranks[:num_checkpoints] # truncate in case length exceeds max
            epochs = epochs_full[:len(ranks)] # truncate in case not finished training

            ax2.set_ylabel("Rank", fontsize=15)
            l2 = ax2.errorbar(epochs, ranks, label=label, color=cmap(i), linestyle=style)
        
        # Construct the Graph
        plt.xlim(1,num_checkpoints*save_skip)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Epoch', fontsize=15)
        
    if plot_rank:
        fig.legend(loc='upper right', prop={'size': 6})
    else:
        ax1.legend(loc='upper right', prop={'size': 6})
        
    filename = "comp_" + plot_name + "_" + parameter_name.replace(" ", "_") + "." + filetype
    plt.savefig(os.path.join(save_directory, filename))
    plt.savefig(os.path.join(save_directory, filename))


# Plot the normalized rank
def plot_rank_normalized(plot_name, parameter_name, parameters, xids, pid, save_directory, filetype, tol=0.001, stable_rank=False, semilog=False):
    if len(parameters) != len(xids): raise ValueError("Parameters and xids must be of same length")
    
    for xid in xids: check_training_progress(xid, pid)
    
    if stable_rank: plot_types = ['rank', 'stable_rank']
    else: plot_types = ['rank']
    
    for plot_type in plot_types:
        for i, xid in enumerate(xids):
            if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))): continue
            
            if plot_type == 'rank':
                ranks = compute_avg_rank(xid, pid, tol)
                style = '-'
                label = parameter_name + ' ' + str(parameters[i])
            elif plot_type == 'stable_rank':
                if os.path.isfile(os.path.join(results_directory, str(xid), str(pid), "avg_stable_rank.txt")):
                    ranks = load_data(xid, pid, 'avg_stable_rank') 
                else: # for old experiments where stable rank not explicitly recorded
                    ranks = compute_avg_stable_rank(xid, pid)
                style = '--'
                label = 'stable rank; ' + parameter_name + ' ' + str(parameters[i])
            
            ranks = ranks[:num_checkpoints]/ranks[0] # truncate in case length exceeds max
            epochs = epochs_full[:len(ranks)] # truncate in case not finished training
            plt.errorbar(epochs, ranks, label=label, color=cmap(i), linestyle=style)
            
            plt.xlim(1, num_checkpoints*save_skip)
            plt.ylim(0, 1)
            if semilog: plt.semilogy()
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel('Epoch', fontsize=15)
            plt.ylabel('Normalized Avg. Rank', fontsize=15)
            plt.legend(loc='lower right', prop={'size':8})
    
    filename = "rank_normalized_" + plot_name + "_" + parameter_name.replace(" ", "_") + "." + filetype
    plt.savefig(os.path.join(save_directory, filename))
    plt.show()


# 3D plot of the singular values across epochs
def plot_singvals_epoch(plot_name, xid, pid, layer, save_directory, filetype):
    check_training_progress(xid, pid)
    if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))): return

    singvals = load_singvals(xid, pid)[layer]

    fig = plt.figure(figsize=(5, 5)) 
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(25, 45)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Index")
    ax.set_zlabel("Singular Value")

    xpos, ypos, zpos, dx, dy, dz = [], [], [], [], [], []

    for i, epoch in enumerate(singvals):
        for j, singval in enumerate(epoch):
            xpos.append(i*save_skip)
            ypos.append(j)
            zpos.append(0)

            dx.append(0.5)
            dy.append(5)
            dz.append(singval)

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, lightsource=matplotlib.colors.LightSource(70, 0))

    filename = "singvals_epoch_" + plot_name + "_layer_" + str(layer) + "." + filetype
    plt.savefig(os.path.join(save_directory, filename))
    plt.show()


# Plot rank and stable rank
def plot_rank(plot_name, parameter_name, parameters, xids, pid, save_directory, filetype, tol=0.001, stable_rank=False, semilog=False, rank_max=500, rank_bound=None, rank_bound_label=None, legend_loc='lower right'):
    if len(parameters) != len(xids): raise ValueError("Parameters and xids must be of same length")
    
    for xid in xids: check_training_progress(xid, pid)
    
    if stable_rank: plot_types = ['rank', 'stable_rank']
    else: plot_types = ['rank']
    
    for plot_type in plot_types:
        for i, xid in enumerate(xids):
            try:
                if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))): continue

                if plot_type == 'rank':
                    ranks = compute_avg_rank(xid, pid, tol)
                    style = '-'
                    label = 'rank ' + parameter_name + ' ' + str(parameters[i])
                elif plot_type == 'stable_rank':
                    if os.path.isfile(os.path.join(results_directory, str(xid), str(pid), "avg_stable_rank.txt")):
                        ranks = load_data(xid, pid, 'avg_stable_rank') 
                    else: # for old experiments where stable rank not explicitly recorded
                        ranks = compute_avg_stable_rank(xid, pid)
                    style = '--'
                    label = 'stable rank; ' + parameter_name + ' ' + str(parameters[i])

                ranks = ranks[:num_checkpoints] # truncate in case length exceeds max
                epochs = epochs_full[:len(ranks)] # truncate in case not finished training
                plt.errorbar(epochs, ranks, label=label, color=cmap(i), linestyle=style)

                plt.xlim(1, num_checkpoints*save_skip)
                plt.ylim(1, rank_max)
                if semilog: plt.semilogy()
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                plt.xlabel('Epoch', fontsize=15)
                plt.ylabel('Average Rank', fontsize=15)
                plt.legend(loc=legend_loc, prop={'size':8})
            except:
                print("Failed to plot experiment:", xid)
                
    if rank_bound is not None:
        yticks = []
        for i, rb in enumerate(rank_bound):
            plt.axhline(rb, color=cmap(i+1), linestyle='--', label='Rank Bound, $\epsilon='+str(rank_bound_label[i])+'$')
            yticks.append(rb)
            
        plt.ylim(min(ranks)/2, rank_max)
        plt.semilogy()
        
        ax = plt.gca()
        yticks += [tick for tick in ax.get_yticks() if min(ranks)/2 <= tick <= rank_max]
        
        print(ax.get_yticks())
        print(yticks)
            
        ax.set_yticks(yticks)
        ax.set_yticklabels(['${:,.2e}$'.format(tick) for tick in yticks])
        plt.legend(loc=legend_loc, prop={'size':14})
        
    filename = "rank_" + plot_name + "_" + parameter_name.replace(" ", "_") + "." + filetype
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, filename))
    plt.show()


# 2D plot of the singular values at the end of training across various hyperparameters
def plot_singvals_param(plot_name, parameter_name, parameters, xids, pid, layer, save_directory, filetype, tol=None):
    if len(parameters) != len(xids): raise ValueError("Parameters and xids must be of same length")
    
    for xid in xids: check_training_progress(xid, pid)
    
    exists=False
    for i, xid in enumerate(xids):
        if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))): continue
        exists=True
        
        style = "-"
        label = parameter_name + ' ' + str(parameters[i])
        
        singvals = load_singvals(xid, pid)[layer][-1]
        plt.errorbar(range(len(singvals)), singvals, label=label, color=cmap(i), linestyle=style, fmt='.')
    
    if not exists: return
    
    if tol is not None:
        plt.errorbar(range(len(singvals)), [tol]*len(singvals), label='tol', color=cmap(len(xids)+1), linestyle='--')
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Index', fontsize=15)
    plt.ylabel('Singular Value', fontsize=15)
    plt.legend(loc='lower left', prop={'size':8})
    plt.semilogy()
    
    filename = "singvals_param_" + plot_name + "_layer_" + str(layer) + "." + filetype
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, filename))
    plt.show()


# 3D plot of the rank by layer over epochs
def plot_rank_by_layer(plot_name, xid, pid, save_directory, filetype, tol=0.001):
    check_training_progress(xid, pid)
    if not os.path.isdir(os.path.join(results_directory, str(xid), str(pid))): return

    fig = plt.figure(figsize=(10,10))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.view_init(30, 0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Layer")
    ax1.set_zlabel("Rank")
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.view_init(30, 60)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Layer")
    ax2.set_zlabel("Rank")
    
    xpos, ypos, zpos, dx, dy, dz = [], [], [], [], [], []
    
    ranks = compute_rank(xid, pid, tol)
    
    for layer_num in range(len(ranks)):
        for epoch in range(len(ranks[layer_num])):
            xpos.append(epoch*save_skip)
            ypos.append(layer_num)
            zpos.append(0)
            
            dx.append(save_skip)
            dy.append(1)
            dz.append(ranks[layer_num, epoch])
            
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, lightsource=matplotlib.colors.LightSource(45, 0))
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz, lightsource=matplotlib.colors.LightSource(45, 0))
    
    filename = "rank_by_layer_" + plot_name + "_depth_" + str(num_layers(xid, pid)) + "." + filetype
    plt.savefig(os.path.join(save_directory, filename))
    plt.show()


# Plots rank and accuracy
def plot_all(plot_name, parameter_name, parameters, xids, pid, save_directory, filetype, tol=0.001, stable_rank=False, rank_max=500, plot_singvals=False, rank_bound=None, rank_bound_label=None, legend_loc='lower right'):
    plot_rank(plot_name, parameter_name, parameters, xids, pid, save_directory, filetype, tol=tol, stable_rank=stable_rank, rank_max=rank_max, rank_bound=rank_bound, rank_bound_label=rank_bound_label, legend_loc=legend_loc)
    if plot_singvals: plot_singvals_param(plot_name, parameter_name, parameters, xids, pid, 3, save_directory, filetype, tol=tol)
    plot_accuracy(plot_name, parameter_name, parameters, xids, pid, save_directory, filetype)
