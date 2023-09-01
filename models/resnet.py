import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    name = "BasicBlock"
    expansion = 1

    def __init__(self, in_planes, planes, activation="relu", stride=1,):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
            
        if type(activation) == type(nn.ReLU()):
            self.activation = F.relu
        elif type(activation) == type(nn.LeakyReLU()):
            self.activation = F.leaky_relu
        else:
            raise Exception("Invalid activation function", activation)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    name = "Bottleneck"
    expansion = 4

    def __init__(self, in_planes, planes, activation="relu", stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        if type(activation) == type(nn.ReLU()):
            self.activation = F.relu
        elif type(activation) == type(nn.LeakyReLU()):
            self.activation = F.leaky_relu
        else:
            raise Exception("Invalid activation function", activation)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, settings, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_input_channels = settings.num_input_channels
        self.activation = settings.activation

        self.num_matrices = 0
        if block.name == "BasicBlock":
            self.num_matrices = sum(num_blocks) * 2 + 1
        elif block.name == "Bottleneck":
            self.num_matrices = sum(num_blocks) * 3 + 1

        self.layers = nn.Sequential()

        self.layers.append(nn.Conv2d(self.num_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(self.activation)

        self._make_layer(block, 64, num_blocks[0], activation=self.activation, stride=1)
        self._make_layer(block, 128, num_blocks[1], activation=self.activation, stride=2)
        self._make_layer(block, 256, num_blocks[2], activation=self.activation, stride=2)
        self._make_layer(block, 512, num_blocks[3], activation=self.activation, stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, activation, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        for stride in strides:
            self.layers.append(block(self.in_planes, planes, activation, stride))
            self.in_planes = planes * block.expansion

    def forward(self, x):
        out = self.layers(x)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(settings):
    return ResNet(BasicBlock, [2, 2, 2, 2], settings)


def ResNet34(settings):
    return ResNet(BasicBlock, [3, 4, 6, 3], settings)


def ResNet50(settings):
    return ResNet(Bottleneck, [3, 4, 6, 3], settings)


def ResNet101(settings):
    return ResNet(Bottleneck, [3, 4, 23, 3], settings)


def ResNet152(settings):
    return ResNet(Bottleneck, [3, 8, 36, 3], settings)


def resnet(settings):
    if settings.resnet_version == 18:
        return ResNet18(settings)  # set to use ResNet18 (for now)
    elif settings.resnet_version == 34:
        return ResNet34(settings)
    elif settings.resnet_version == 50:
        return ResNet50(settings)
    elif settings.resnet_version == 101:
        return ResNet101(settings)
    elif settings.resnet_version == 152:
        return ResNet152(settings)
    else:
        raise Exception("Invalid ResNet architecture.")
