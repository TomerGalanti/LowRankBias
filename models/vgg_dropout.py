import torch.nn as nn
import torch

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
vgg_name = 'VGG16'


def prev_dim(input_dimensions):
    if len(input_dimensions) == 0:
        return 32  # First layer has dimension 32

    for x in reversed(input_dimensions):
        if x is not None: return x


def get_num_layers(layer_arch):
    layer_count = 0
    for x in layer_arch:
        if type(x) is int: layer_count += 1

    return layer_count


class VGGDropout(nn.Module):
    def __init__(self, settings):
        super(VGGDropout, self).__init__()

        self.num_input_channels = settings.num_input_channels
        self.num_matrices = get_num_layers(cfg[vgg_name])
        layers = nn.Sequential()

        self.input_dimensions = []
        for x in cfg[vgg_name]:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.input_dimensions += [int(prev_dim(self.input_dimensions) / 2)]
            else:
                layers.append(nn.Conv2d(self.num_input_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(inplace=True))
                self.num_input_channels = x
                self.input_dimensions += [prev_dim(self.input_dimensions), None, None]
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))  # WHAT IS THIS DOING?? <<<<<<
        self.input_dimensions += [self.num_input_channels]

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(512, settings.num_output_classes)

        # self.softmax = nn.Softmax()

        self.layers = layers

    def forward(self, x):

        output = self.layers(x)
        output = output.view(output.shape[0], -1)
        output = self.dropout(output)
        output = self.fc(output)
        # output = self.softmax(output)

        return output


def vgg_dropout(settings):
    return VGGDropout(settings)
