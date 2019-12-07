import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class MelDiscriminatorNet(nn.Module):
    """
    Discriminator for GAN training.
    Based on MelNET discriminator.

    Code borrowed from https://github.com/seungwonpark/melgan/blob/master/model/discriminator.py
    """

    def __init__(self):
        super().__init__()
        layers = [
            conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.PReLU(),
            conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            nn.PReLU(),
            conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            nn.PReLU(),
            conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
            nn.PReLU(),
            conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
            nn.PReLU(),
            conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.PReLU(),
            conv1d(1024, 1, kernel_size=3, stride=1, padding=1),
        ]
        self.discriminator = nn.Sequential(*layers)

    def forward(self, input_t):
        batch_size = input_t.shape[0]
        input_t = input_t.view(batch_size, 1, -1)
        return self.discriminator(input_t)

    def freeze(self):
        self._set_requires_grad(False)

    def unfreeze(self):
        self._set_requires_grad(True)

    def _set_requires_grad(self, val):
        for param in self.parameters():
            param.requires_grad = val

        for layer in self.discriminator:
            for param in layer.parameters():
                param.requires_grad = val


def conv1d(in_channels, out_channels, **kwargs):
    return weight_norm(nn.Conv1d(in_channels, out_channels, **kwargs))
