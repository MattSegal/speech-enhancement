import torch
import numpy as np


class DummyNet(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_cuda):
        super().__init__()
        self.use_cuda = use_cuda
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dummy = torch.nn.Parameter(torch.tensor([1.0]))

    def forward(self, input_t):
        assert input_t.shape == self.input_shape
        t = torch.Tensor(np.random.random(self.output_shape))
        return t.cuda() if self.use_cuda else t.cpu()
