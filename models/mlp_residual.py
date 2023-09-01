
import torch.nn as nn
from models.modules import EmbSeq, FCBlock
import math

class MLPResidual(nn.Module):

    def __init__(self, settings):
        super().__init__()

        self.num_input_channels = settings.num_input_channels
        self.width = settings.width
        self.depth = settings.depth
        self.num_embs = self.depth
        self.num_matrices = 2 * self.depth
        self.alpha = settings.alpha

        layers = []

        width = self.num_input_channels*32*32

        layers += [(nn.Linear(width, self.width), False),
                   (nn.BatchNorm1d(self.width), False)]

        layers += self.depth*[(FCBlock(self.width, self.width, self.alpha), True)]

        self.layers = EmbSeq(layers)

        self.fc = nn.Linear(self.width, settings.num_output_classes)

    def forward(self, x):

        output = x.view(x.shape[0], -1)
        output, embeddings = self.layers(output, [])
        output = self.fc(output)

        return output, embeddings


def mlp_residual(settings):
    return MLPResidual(settings)
