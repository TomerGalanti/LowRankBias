import torch
import torch.nn as nn

@torch.no_grad()
def eval_dist(list_weights1, list_weights2, normalize=False):

    avg_dist = 0
    for i in range(len(list_weights1)):
        dist = ((torch.norm(list_weights1[i] - list_weights2[i])).item())
        if normalize: 
            norm = torch.norm(list_weights1[i]).item()
            if norm != 0: dist = dist / norm
        avg_dist += dist / len(list_weights1)

    return avg_dist


@torch.no_grad()
def get_weights(model, include_top_layer=True):

    list_weights = []

    for i,layer in enumerate(model.layers):

        if type(layer) == nn.Linear:
            list_weights += [layer.weight.data.clone()]
        if type(layer) == nn.Conv2d:
            list_weights += [layer.weight.data.clone()]

    if include_top_layer:
        list_weights += [model.fc.weight]

    return list_weights



