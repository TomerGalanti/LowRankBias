import torch
import torch.nn as nn
from torch import Tensor

class FCBlock(nn.Module):
    def __init__(
            self,
            input_dim,
            width,
            alpha=1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.width = width
        self.fc1 = nn.Linear(input_dim, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        output = identity + self.alpha*x
        return output


class ConvBlock(nn.Module):
    def __init__(
            self,
            width,
            alpha=1,
    ) -> None:
        super().__init__()

        self.width = width

        self.conv1 = nn.Conv2d(self.width, self.width, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(self.width, self.width, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        z = self.relu(x)
        x = self.conv2(z)
        x = self.bn2(x)
        output = identity + self.alpha*x
        return output