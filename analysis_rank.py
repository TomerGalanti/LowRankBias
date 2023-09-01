from models import modules
from models import resnet

from utils import get_training_dataloader

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

@torch.no_grad()
def eval_ranks(model, normalize=True, tol=0.001, device='cpu'):
    ranks = []
    for i, layer in enumerate(model.layers):
        if type(layer) == nn.Linear:
            weights = layer.weight.to(device)
            if normalize: weights = weights / torch.norm(weights)
            ranks += [torch.linalg.matrix_rank(weights, tol=tol).item()]

        elif type(layer) == nn.Conv2d:
            weights = layer.weight.to(device)
            if normalize: weights = weights / torch.norm(weights)

            ranks += [eval_rank_conv_prime(weights, tol)]

        elif type(layer) == modules.FCBlock:
            weights1 = layer.fc1.weight.to(device)
            weights2 = layer.fc2.weight.to(device)
            if normalize:
                weights1 = weights1 / torch.norm(weights1)
                weights2 = weights2 / torch.norm(weights2)
            ranks += [torch.linalg.matrix_rank(weights1, tol=tol).item()]
            ranks += [torch.linalg.matrix_rank(weights2, tol=tol).item()]

        elif type(layer) == modules.ConvBlock:
            weights1 = layer.conv1.weight.to(device)
            weights2 = layer.conv2.weight.to(device)
            if normalize:
                weights1 = weights1 / torch.norm(weights1)
                weights2 = weights2 / torch.norm(weights2)
            ranks += [eval_rank_conv(weights1, (model.input_dimensions[i], model.input_dimensions[i]), tol)]
            ranks += [eval_rank_conv(weights2, (model.input_dimensions[i], model.input_dimensions[i]), tol)]

        elif type(layer) == resnet.BasicBlock:
            weights1 = layer.conv1.weight.to(device)
            weights2 = layer.conv2.weight.to(device)
            if normalize:
                weights1 = weights1 / torch.norm(weights1)
                weights2 = weights2 / torch.norm(weights2)
            ranks += [eval_rank_conv_prime(weights1, tol)]
            ranks += [eval_rank_conv_prime(weights2, tol)]

        elif type(layer) == resnet.Bottleneck:
            weights1 = layer.conv1.weight.to(device)
            weights2 = layer.conv2.weight.to(device)
            weights3 = layer.conv3.weight.to(device)
            if normalize:
                weights1 = weights1 / torch.norm(weights1)
                weights2 = weights2 / torch.norm(weights2)
                weights3 = weights3 / torch.norm(weights3)
            ranks += [eval_rank_conv_prime(weights1, tol)]
            ranks += [eval_rank_conv_prime(weights2, tol)]
            ranks += [eval_rank_conv_prime(weights3, tol)]

    return ranks

@torch.no_grad()
def eval_singvals(model, normalize=True, device='cpu'):
    singvals = []
    for i, layer in enumerate(model.layers):
        if type(layer) == nn.Linear:
            weights = layer.weight.to(device)
            if normalize: weights = weights / torch.norm(weights)

            singular_vals = torch.flatten(torch.linalg.svdvals(weights)).tolist()
            singvals += [singular_vals]
        elif type(layer) == nn.Conv2d:
            weights = layer.weight.to(device)
            if normalize: weights = weights / torch.norm(weights)

            kernel = weights.view(weights.shape[0], -1)
            singular_vals = torch.flatten(torch.linalg.svdvals(kernel)).tolist()
            singvals += [singular_vals]

        elif type(layer) == resnet.BasicBlock:
            weights1 = layer.conv1.weight.to(device)
            weights2 = layer.conv2.weight.to(device)
            if normalize:
                weights1 = weights1 / torch.norm(weights1)
                weights2 = weights2 / torch.norm(weights2)

            kernel1 = weights1.view(weights1.shape[0], -1)
            singular_vals1 = torch.flatten(torch.linalg.svdvals(kernel1)).tolist()
            singvals += [singular_vals1]
            kernel2 = weights2.view(weights2.shape[0], -1)
            singular_vals2 = torch.flatten(torch.linalg.svdvals(kernel2)).tolist()
            singvals += [singular_vals2]

        elif type(layer) == resnet.Bottleneck:
            weights1 = layer.conv1.weight.to(device)
            weights2 = layer.conv2.weight.to(device)
            weights3 = layer.conv2.weight.to(device)
            if normalize:
                weights1 = weights1 / torch.norm(weights1)
                weights2 = weights2 / torch.norm(weights2)
                weights3 = weights2 / torch.norm(weights2)

            kernel1 = weights1.view(weights1.shape[0], -1)
            singular_vals1 = torch.flatten(torch.linalg.svdvals(kernel1)).tolist()
            singvals += [singular_vals1]
            kernel2 = weights2.view(weights2.shape[0], -1)
            singular_vals2 = torch.flatten(torch.linalg.svdvals(kernel2)).tolist()
            singvals += [singular_vals2]
            kernel3 = weights3.view(weights3.shape[0], -1)
            singular_vals3 = torch.flatten(torch.linalg.svdvals(kernel3)).tolist()
            singvals += [singular_vals3]

    return singvals

# returns rank_bound_C, where:
# rank_bound_C is ||grad_{Vk} L||*||Vk||/(|g_k grad_{gk} L|)
def eval_rank_bound_C(model, settings, device='cuda'):
    batch_size = settings.bound_batch_size
    num_batches = settings.bound_num_batches
    weight_decay = settings.weight_decay

    train_loader = get_training_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=batch_size,
        shuffle=settings.shuffle,
        rnd_aug=settings.rnd_aug,
        num_classes=settings.num_output_classes,
        bound_num_batches=num_batches
    )

    if settings.loss == 'CE': loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE': loss_function = nn.MSELoss()

    print("Calculating rank bound C:")

    rank_bound_layer = []
    for i, layer in enumerate(tqdm(model.layers)):
        if type(layer) != nn.Linear:
            continue

        min_rank_bound = float('inf')
        for batch_index, (images, labels) in enumerate(train_loader):
            if settings.device == 'cuda':
                labels = labels.cuda()
                images = images.cuda()
            
            outputs = model(images)
            if settings.loss == 'CE':
                loss = loss_function(outputs, labels)
            elif settings.loss == 'MSE':
                loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

            weight_v = layer.weight_v.to(device)
            weight_v_norm = torch.linalg.norm(weight_v, ord='fro')

            model.zero_grad()
            grad_v = torch.autograd.grad(loss, weight_v, retain_graph=True)[0]
            grad_v_norm = torch.linalg.norm(grad_v, ord=2) # spectral norm

            model.zero_grad()
            weight_g = layer.weight_g.to(device)
            grad_g = torch.autograd.grad(loss, weight_g)[0]
            
            denom = torch.abs(weight_g * grad_g)

            rank_bound = (grad_v_norm * weight_v_norm)/(denom)
            min_rank_bound = min(rank_bound, min_rank_bound)

        rank_bound_layer += [min_rank_bound.item()]

    print(rank_bound_layer)
    return rank_bound_layer

# returns rank_bound_B, where:
# rank_bound_B is min_batch ||grad_{Vi} L(f_w)|| * ||Vi|| / | (1/B) * sum_{j} (<dloss_j/df_j, f_j>) |
def eval_rank_bound_B(model, settings, device='cuda'):
    batch_size = settings.bound_batch_size
    num_batches = settings.bound_num_batches
    weight_decay = settings.weight_decay

    train_loader = get_training_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=batch_size,
        shuffle=settings.shuffle,
        rnd_aug=settings.rnd_aug,
        num_classes=settings.num_output_classes,
        bound_num_batches=num_batches
    )

    if settings.loss == 'CE': loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE': loss_function = nn.MSELoss()

    print("Calculating rank bound B:")

    rank_bound_layer = []
    for i, layer in enumerate(tqdm(model.layers)):
        if type(layer) != nn.Linear: # change to weight norm?
            continue

        min_rank_bound = float('inf')
        for batch_index, (images, labels) in enumerate(train_loader):
            if settings.device == 'cuda':
                labels = labels.cuda()
                images = images.cuda()
            
            outputs = model(images)
            if settings.loss == 'CE':
                loss = loss_function(outputs, labels)
            elif settings.loss == 'MSE':
                loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

            weight_v = layer.weight_v.to(device)
            weight_v_norm = torch.linalg.norm(weight_v, ord='fro')

            model.zero_grad()
            grad = torch.autograd.grad(loss, weight_v)[0]
            grad_norm = torch.linalg.norm(grad, ord=2) # spectral norm

            avg_dL_batch = 0
            for index in range(len(images)):
                image = images[index]
                label = labels[index]
                output = outputs[index]
                
                if settings.loss == 'CE':
                    loss = loss_function(output, label)
                elif settings.loss == 'MSE':
                    loss = loss_function(output, F.one_hot(label, settings.number_output_classes).float())

                model.zero_grad()
                partials = torch.autograd.grad(loss, output)[0]

                # take the dot product of the gradient and output
                partial_dot = torch.dot(partials, output)

                avg_dL_batch += partial_dot

            avg_dL_batch /= len(images)
            avg_dL_batch = torch.abs(avg_dL_batch)
            
            rank_bound = (grad_norm * weight_v_norm)/(avg_dL_batch)
            min_rank_bound = min(rank_bound, min_rank_bound)

        rank_bound_layer += [min_rank_bound.item()]

    print(rank_bound_layer)
    return rank_bound_layer

# returns rank_bound_A, where
# rank_bound_A is ||grad L(f(w))||/(||W^k||)
# and grad L(f(w)) is the average gradient of the non-regularized loss across all minibatches
def eval_rank_bound_A(model, settings, device='cuda'):
    batch_size = settings.bound_batch_size
    weight_decay = settings.weight_decay

    train_loader = get_training_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=batch_size,
        shuffle=settings.shuffle,
        rnd_aug=settings.rnd_aug,
        num_classes=settings.num_output_classes,
    )

    if settings.loss == 'CE': loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE': loss_function = nn.MSELoss()

    print("Calculating rank bound A:")

    rank_bound = []
    for i, layer in enumerate(tqdm(model.layers)):
        if type(layer) != nn.Linear:
            continue

        grad = 0
        num_batches = 0
        for batch_index, (images, labels) in enumerate(tqdm(train_loader,leave=False)):
            num_batches+=1
            # compute gradient of loss for sample
            model.zero_grad()

            if settings.device == 'cuda':
                labels = labels.cuda()
                images = images.cuda()
            
            outputs = model(images)
            if settings.loss == 'CE':
                loss = loss_function(outputs, labels)
            elif settings.loss == 'MSE':
                loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

            weights = layer.weight.to(device)
            weight_norm = torch.linalg.norm(weights,ord='fro')

            grad += torch.autograd.grad(loss, weights)[0]

        grad /= num_batches
        grad_norm = torch.linalg.norm(grad,ord=2) # spectral norm

        batch_rank_bound = grad_norm/weight_norm

        rank_bound += [batch_rank_bound.item()]

    return rank_bound


# returns rank_bound_full, where
# rank_bound_full is ||grad L(f(w))||/(2*\lambda*||W||)
# and grad L(f(w)) is the average gradient of the regularized loss across all minibatches
def eval_rank_bound_full(model, settings, device='cuda'):
    batch_size = settings.bound_batch_size
    weight_decay = settings.weight_decay

    train_loader = get_training_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=batch_size,
        shuffle=settings.shuffle,
        rnd_aug=settings.rnd_aug,
        num_classes=settings.num_output_classes,
    )

    if settings.loss == 'CE': loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE': loss_function = nn.MSELoss()

    print("Calculating rank bound full:")

    rank_bound = []
    for i, layer in enumerate(tqdm(model.layers)):
        if type(layer) != nn.Linear:
            continue

        grad = 0
        num_batches = 0
        for batch_index, (images, labels) in enumerate(tqdm(train_loader,leave=False)):
            num_batches+=1
            # compute gradient of loss for sample
            model.zero_grad()

            if settings.device == 'cuda':
                labels = labels.cuda()
                images = images.cuda()
            
            outputs = model(images)
            if settings.loss == 'CE':
                loss = loss_function(outputs, labels)
            elif settings.loss == 'MSE':
                loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

            weights = layer.weight.to(device)
            weight_norm = torch.linalg.norm(weights,ord='fro')

            loss += weight_decay*(weight_norm**2)

            grad += torch.autograd.grad(loss, weights)[0]

        grad /= num_batches
        grad_norm = torch.linalg.norm(grad,ord=2) # spectral norm

        print("Grad norm, layer "+str(i)+":", grad_norm)

        batch_rank_bound = (1/(2*weight_decay))*grad_norm*(1/weight_norm)

        rank_bound += [batch_rank_bound.item()]

    return rank_bound

# returns rank_bound and new_rank_bound, where:
# rank_bound is min_batch ||grad L(f(w))||/(2*\lambda*||W||)
# new_rank_bound is min_batch ||grad L(f(w))||/(avg [dL(f(x))/df(x) for x in the batch]), and dL(f(x))/df(x) is the average of the partial derivatives of f(x)
def eval_rank_bound(model, settings, device='cuda'):
    batch_size = settings.bound_batch_size
    num_batches = settings.bound_num_batches
    weight_decay = settings.weight_decay

    train_loader = get_training_dataloader(
        settings.dataset_name,
        settings.mean,
        settings.std,
        num_workers=2,
        batch_size=batch_size,
        shuffle=settings.shuffle,
        rnd_aug=settings.rnd_aug,
        num_classes=settings.num_output_classes,
        bound_num_batches=num_batches
    )

    if settings.loss == 'CE': loss_function = nn.CrossEntropyLoss()
    elif settings.loss == 'MSE': loss_function = nn.MSELoss()

    print("Calculating rank bound:")

    rank_bound = []
    rank_bound_new = []
    for i, layer in enumerate(tqdm(model.layers)):
        if type(layer) != nn.Linear:
            continue

        min_rank_bound = float('inf')
        min_rank_bound_new = float('inf')
        for batch_index, (images, labels) in enumerate(train_loader):
            if settings.device == 'cuda':
                labels = labels.cuda()
                images = images.cuda()
            
            outputs = model(images)
            if settings.loss == 'CE':
                loss = loss_function(outputs, labels)
            elif settings.loss == 'MSE':
                loss = loss_function(outputs, F.one_hot(labels, settings.num_output_classes).float())

            weights = layer.weight.to(device)
            weight_norm = torch.linalg.norm(weights,ord='fro')

            loss += weight_decay*(weight_norm**2)

            model.zero_grad()
            grad = torch.autograd.grad(loss, weights)[0]
            grad_norm = torch.linalg.norm(grad,ord=2) # spectral norm

            # CALCULATE RANK BOUND
            batch_rank_bound = grad_norm/(2*weight_decay*weight_norm)
            min_rank_bound = min(batch_rank_bound,min_rank_bound)

            # CALCULATE NEW RANK BOUND
            avg_dL_batch = 0
            for index in range(len(images)):
                image = images[index]
                label = labels[index]
                output = outputs[index]
                
                if settings.loss == 'CE':
                    loss = loss_function(output, label)
                elif settings.loss == 'MSE':
                    loss = loss_function(output, F.one_hot(label, settings.number_output_classes).float())
                loss += weight_decay*(weight_norm**2)

                model.zero_grad()
                partials = torch.autograd.grad(loss, output)
                
                avg_partial = torch.mean(torch.abs(partials[0]))

                avg_dL_batch += avg_partial

            avg_dL_batch/=len(images)
            
            batch_rank_bound_new = grad_norm/avg_dL_batch
            min_rank_bound_new = min(batch_rank_bound_new,min_rank_bound_new)

        rank_bound += [min_rank_bound.item()]
        rank_bound_new += [min_rank_bound_new.item()]

    print("rank_bound_new:",rank_bound_new)
        
    return rank_bound, rank_bound_new

@torch.no_grad()
def eval_stable_ranks(singvals):
    singvals = np.array(singvals)
    stable_ranks = []

    for layer in singvals:
        frobenius_squared = 0
        spectral_squared = 0
        for singval in layer:
            frobenius_squared += singval ** 2
            if singval ** 2 > spectral_squared: spectral_squared = singval ** 2
        if spectral_squared == 0: stable_ranks.append(0)
        else: stable_ranks.append(frobenius_squared/spectral_squared)

    return stable_ranks

# Computes the rank of the matrix representing the linear transformation of the convolutional layer
@torch.no_grad()
def eval_rank_conv(kernel, input_shape, tol):
    transforms = torch.fft.fft2(torch.permute(kernel, (2,3,1,0)),
                                input_shape, dim=[0, 1])
    singular_vals = torch.flatten(torch.linalg.svdvals(transforms))
    rank = len([s for s in singular_vals if abs(s) > tol])

    return rank

# Computes the rank of the matrix whose rows consist of the flattened kernel weights
@torch.no_grad()
def eval_rank_conv_prime(kernel, tol):
    kernel = kernel.view(kernel.shape[0], -1)
    rank = torch.linalg.matrix_rank(kernel, tol=tol).item()

    return rank

# Computes the frobenius norm of the weight matrices of the network
@torch.no_grad()
def eval_f_norm(model, architecture, device='cpu'):
    if architecture not in ['mlp', 'resnet', 'vgg']:
        return None

    norms = []
    for i, layer in enumerate(model.layers):
        if type(layer) == nn.Linear:
            weights = layer.weight.to(device)
            norms.append(torch.linalg.norm(weights).item())
        elif type(layer) == nn.Conv2d:
            weights = layer.weight.to(device)
            norms.append(torch.linalg.norm(weights).item())
        elif type(layer) == resnet.BasicBlock:                                   
            weights1 = layer.conv1.weight.to(device)
            weights2 = layer.conv2.weight.to(device)
            norms.append(torch.linalg.norm(weights1).item())
            norms.append(torch.linalg.norm(weights2).item())
        else:
            continue

    return norms

# Computes the complexity of the network, if supported.
@torch.no_grad()
def eval_complexity(model, architecture, device='cpu'):
    if architecture not in ['mlp', 'convnet']:
        return None

    rho = 1
    for i, layer in enumerate(model.layers):
        if type(layer) == nn.Linear:
            weights = layer.weight.to(device)
            rho *= torch.linalg.norm(weights).item()
        elif type(layer) == nn.Conv2d:
            weights = layer.weight.to(device)
            rho *= torch.linalg.norm(weights).item()
            rho *= model.output_dimensions[i]
        elif type(layer) == nn.ReLU:
            continue
        else:
            print("WARNING: Unable to compute Complexity:", type(layer))
            return None

    return rho


@torch.no_grad()
def eval_dist(list_weights1, list_weights2, lr):

    dist = 0
    for i in range(len(list_weights1)):
        dist += ((torch.norm(list_weights1[i] - list_weights2[i]) / lr).item())

    return dist / (i+1)
