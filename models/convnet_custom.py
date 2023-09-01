import torch.nn as nn
import torch

class ConvNetCustom(nn.Module):
    def __init__(self, settings):
        super(ConvNetCustom, self).__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.num_matrices = self.depth = settings.depth
        self.activation = settings.activation
        self.bn = settings.bn

        self.custom_layer = settings.custom_layer-1 # 1-indexed
        self.kernel_dim = settings.kernel_dim
        self.stride_length = settings.stride_length
        self.padding = settings.padding
        self.custom_width = settings.custom_width

        layers = nn.Sequential()

        self.input_dimensions = []
        width = self.num_input_channels # initial width
        prev_output_dim = 32 # first layer has dimension 32

        for i in range(self.depth):
            self.input_dimensions += [prev_output_dim, None, None]

            if i == self.custom_layer:
                layers.append(nn.Conv2d(width, self.custom_width, self.kernel_dim, self.stride_length, self.padding))
                if self.bn: layers.append(nn.BatchNorm2d(self.custom_width))
                prev_output_dim = \
                    int((prev_output_dim+2*self.padding-self.kernel_dim)/self.stride_length+1) # round down
                width = self.custom_width

            else:
                layers.append(nn.Conv2d(width,self.width,3,1,1))
                if self.bn: layers.append(nn.BatchNorm2d(self.width))
                width = self.width

            layers.append(self.activation)

        self.layers = layers
        self.fc = nn.Linear(self.width * prev_output_dim * prev_output_dim, settings.num_output_classes)

    def forward(self, x):
        output = self.layers(x)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)

        return output


def convnet_custom(settings):
    return ConvNetCustom(settings)
