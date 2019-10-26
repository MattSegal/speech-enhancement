import torch
from torch import nn


class Interpolation1d(nn.Module):
    """
    Trainable 2x upsampling layer.
    Uses same padding strategy
    """

    def __init__(self, num_features):
        super().__init__()
        # self.alpha = nn.Parameter(torch.tensor([0.0]))
        # self.beta = nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_t):
        # norm_t = self.batch_norm(input_t)
        # return self.alpha * input_t + self.beta * norm_t
