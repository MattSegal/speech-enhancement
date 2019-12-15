import torch
import numpy as np


class DummyNet(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dummy = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_t):
        assert input_t.shape == self.input_shape
        return torch.Tensor(np.random.random(self.output_shape))
