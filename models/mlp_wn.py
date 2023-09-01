
import torch.nn as nn
import torch
import torch.nn.utils.weight_norm as weight_norm
from collections import OrderedDict

"""
NOTE: In the train.py, weight_decay is only applied to the 'fc' parameter.
"""
class MLP_WN(nn.Module):

    def __init__(self, settings):
        super().__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.num_matrices = self.depth = settings.depth
        self.activation = settings.activation
        self.bias = settings.bias

        layers = nn.Sequential()

        width = self.num_input_channels*32*32
        for i in range(self.depth):
            m = weight_norm(nn.Linear(width, self.width, bias=self.bias),dim=None)
            layers.append(m)
            layers.append(self.activation)

            width = self.width

        self.layers = layers
        self.fc = nn.Linear(width, settings.num_output_classes, bias=self.bias)

    def forward(self, x):
        output = x.view(x.shape[0], -1) # flattens the data to dim [batch_size, width]

        output = self.layers(output)
        output = self.fc(output)

        return output


def mlp_wn(settings):
    return MLP_WN(settings)
