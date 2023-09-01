
import torch.nn as nn
import torch

class ConvNet(nn.Module):
    def __init__(self, settings):
        super(ConvNet, self).__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.num_matrices = self.depth = settings.depth 
        self.activation = settings.activation
        self.bn = settings.bn

        layers = nn.Sequential()

        self.input_dimensions = [32]
        self.output_dimensions = [16]

        layers.append(nn.Conv2d(self.num_input_channels, self.width, 2, 2))
        if self.bn:
            layers.append(nn.BatchNorm2d(self.width))
            self.input_dimensions.append(None)
            self.output_dimensions.append(None)

        self.input_dimensions += [16, None]
        self.output_dimensions += [8, None]
        layers.append(nn.Conv2d(self.width, self.width, 2, 2))
        if self.bn:
            layers.append(nn.BatchNorm2d(self.width))
            self.input_dimensions.append(None)
            self.output_dimensions.append(None)
        layers.append(self.activation)

        for i in range(self.depth):
            self.input_dimensions += [8, None]
            self.output_dimensions += [8, None]

            layers.append(nn.Conv2d(self.width, self.width, 3, 1, 1))
            if self.bn:
                layers.append(nn.BatchNorm2d(self.width))
                self.input_dimensions.append(None)
                self.output_dimensions.append(None)
            layers.append(self.activation)

        self.layers = layers

        self.fc = nn.Linear(self.width*8*8, settings.num_output_classes)

    def forward(self, x):

        output= self.layers(x)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)

        return output

def convnet(settings):
    return ConvNet(settings)
