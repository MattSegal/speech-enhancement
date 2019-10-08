import torch
import torch.nn as nn


class SpeechDenoiseNet(nn.Module):
    """
    Convolutional network used to denoise human speech in audio.
    """

    def __init__(self):

        ConvLayer(in_channels=1, out_channels=32, dilation=1)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**1)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**2)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**3)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**4)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**5)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**6)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**7)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**8)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**9)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**10)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**11)
        ConvLayer(in_channels=32, out_channels=32, dilation=2**12)
        ConvLayer(in_channels=32, out_channels=32, dilation=1)

        self.final_conv = nn.Conv1d(
            in_channels=32
            out_channels=1,
            kernel_size=1,
            bias=True,
        )


    def forward(self, input_t):
        """
        Input has shape (batch_size, 1, audio_length,) 
        Output has shape (batch_size, 1, audio_length,) 
        """

        # Each intermediate layer is a 2-dimensional tensor of
        # dimensionality N ×W, where W is the number of feature
        # maps in each layer. (We set W = 64.)
        # (batch_size, 64, audio_length,) ?

        # The content of each intermediate layer is computed from
        # the previous layer via a dilated convolution with
        #  3 × 1 convolutional kernels [26] followed by an
        #  adaptive normalization (see below) and a
        # pointwise nonlinear leaky rectified linear unit (LReLU) [28]

        # We zero-pad all layers so that
        # their “effective” length is constant at N

        # we increase the dilation factor exponentially with depth from 2^0
        # for the 1st intermediate layer to 2^12 for the 13th one.
        # We do not use dilation for the 14th and last one


class ConvLayer(nn.Module):
    """
    Single convolutional unit for the speech denoising network.
        Input tensor: (batch_size, in_channels, length)
        Output tensor: (batch_size, out_channels, length)
    """

    def __init__(self, in_channels, out_channels, dilation):
        """
        Setup the layer.
            in_channels: number of input channels to be convoluted
            out_channels: number of output channels to be produced

        """
        super().__init__()
        # Configure padding so the input and output sizes are the same.
        # You could just do the padding manually with pytorch
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            kernel_size=3,
            padding=1,  # Need to check how this interacts with dilation
            bias=False,  # No bias when using batch norm
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        # Apply Kaiming initialization to convolutional weights
        nn.init.kaiming_normal_(self.conv.weight)
        # Adaptive norm parameters
        self.adapt_alpha = nn.Parameter(torch.tensor([1.0]))
        self.adapt_beta = nn.Parameter(torch.tensor([0.0]))

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        conv_t = self.conv(input_t)
        relu_t = self.lrelu(conv_t)
        norm_t = self.batch_norm(relu_t)
        # Apply adaptive normalization (Fast Image Processing with Fully-Convolutional Networks)
        adapt_t = self.adapt_alpha * relu_t + self.adapt_beta * norm_t
        return adapt_t
