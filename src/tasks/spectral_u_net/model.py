import torch
from torch import nn
from torch.nn.utils import weight_norm


class SpectralUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvLayer(2, 2, kernel=1)

    def forward(self, input_t):
        return self.conv(input_t)


class ConvLayer(nn.Module):
    """
    Single convolutional layer with nonlinear output
    """

    def __init__(self, in_channels, out_channels, kernel, nonlinearity=nn.PReLU):
        super().__init__()
        self.nonlinearity = nonlinearity()
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            padding=kernel // 2,  # Same padding
            bias=True,
        )
        self.conv = weight_norm(conv)
        # Apply Kaiming initialization to convolutional weights
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        acts = self.conv(input_t)
        return self.nonlinearity(acts)
