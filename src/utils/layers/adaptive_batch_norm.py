import torch
from torch import nn


class AdaptiveBatchNorm1d(nn.Module):
    """
    Apply adaptive batch normalization
    as defined in Fast Image Processing with Fully-Convolutional Networks
    """

    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.alpha = nn.Parameter(torch.tensor([0.0]))
        self.beta = nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_t):
        norm_t = self.batch_norm(input_t)
        return self.alpha * input_t + self.beta * norm_t
