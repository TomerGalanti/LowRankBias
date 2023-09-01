
import torch.nn as nn
from models.modules import ConvBlock
import torch

class ConvResNet(nn.Module):
    def __init__(self, settings):
        super(ConvResNet, self).__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.depth = settings.depth
        self.num_matrices = 2*self.depth
        self.alpha = settings.alpha
        self.activation = settings.activation

        layers = nn.Sequential()

        self.input_dimensions = [32, None] # NEW

        layers.append(nn.Conv2d(self.num_input_channels, self.width, 2, 2))
        layers.append(nn.BatchNorm2d(self.width))

        self.input_dimensions += [16, None, None] # NEW

        layers.append(nn.Conv2d(self.width, self.width, 2, 2))
        layers.append(nn.BatchNorm2d(self.width))
        layers.append(self.activation)

        for i in range(self.depth):
            self.input_dimensions += [8, None]  # NEW

            layers.append(ConvBlock(self.width, self.alpha))
            layers.append(self.activation)

        self.layers = layers

        self.fc = nn.Linear(self.width*8*8, settings.num_output_classes)

    def forward(self, x):

        output= self.layers(x)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)

        return output

def convresnet(settings):
    return ConvResNet(settings)