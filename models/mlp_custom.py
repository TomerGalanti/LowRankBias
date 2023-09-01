
import torch.nn as nn
import torch
from collections import OrderedDict

class MLPCustom(nn.Module):

    def __init__(self, settings):
        super().__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.num_matrices = self.depth = settings.depth
        self.activation = settings.activation
        self.bn = settings.bn
        self.bias = settings.bias
        self.custom_layer = settings.custom_layer-1 # 1-indexed
        self.custom_width = settings.custom_width

        layers = nn.Sequential()

        width = self.num_input_channels*32*32
        for i in range(self.depth):
            if i == self.custom_layer:
                layers.append(nn.Linear(width, self.custom_width, bias=self.bias))
                if self.bn: layers.append(nn.BatchNorm1d(self.custom_width))
                width = self.custom_width
            else:
                layers.append(nn.Linear(width, self.width, bias=self.bias))
                if self.bn: layers.append(nn.BatchNorm1d(self.width))
                width = self.width

            layers.append(self.activation)

        self.layers = layers
        self.fc = nn.Linear(width, settings.num_output_classes, bias=self.bias)

    def forward(self, x):
        output = x.view(x.shape[0], -1) # flattens the data to dim [batch_size, width]

        output = self.layers(output)
        output = self.fc(output)

        return output


def mlp_custom(settings):
    return MLPCustom(settings)
