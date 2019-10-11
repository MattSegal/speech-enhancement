import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

# epoch / segments / samples / batch / GPU memory
# 30s / 4 /500 / 32 / ~7GB
# 16s / 5 /500 / 32 / ~4GB
# 16s / 5 /500 / 48 / ~6GB
# 16s / 6 /500 / 32 / ~6GB

GRAD_CHECKPOINT_SEGMENTS = 5
NUM_INNER_CONVS = 12
CHANNELS = 64


class SpeechDenoiseNet(nn.Module):
    """
    Convolutional network used to denoise human speech in audio.
    """

    def __init__(self):
        super().__init__()

        input_conv = ConvLayer(in_channels=1, out_channels=CHANNELS, dilation=1)
        inner_convs = [
            ConvLayer(in_channels=CHANNELS, out_channels=CHANNELS, dilation=2 ** i)
            for i in range(1, NUM_INNER_CONVS + 1)
        ]
        final_inner_conv = ConvLayer(
            in_channels=CHANNELS, out_channels=CHANNELS, dilation=1
        )
        output_conv = nn.Conv1d(
            in_channels=CHANNELS, out_channels=1, kernel_size=1, bias=True
        )
        conv_layers = [input_conv, *inner_convs, final_inner_conv, output_conv]
        self.convs = nn.Sequential(*conv_layers)

    def forward(self, input_t):
        """
        Input has shape (batch_size, 1, audio_length,) 
        Output has shape (batch_size, audio_length,) 
        """
        batch_size = input_t.shape[0]
        assert input_t.shape[1] == 1
        audio_length = input_t.shape[2]
        # Use gradient checkpointing to save GPU memory.
        modules = [m for m in self.convs._modules.values()]
        conv_t = checkpoint_sequential(modules, GRAD_CHECKPOINT_SEGMENTS, input_t)
        return conv_t.squeeze(dim=1)


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
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            dilation=dilation,
            padding=dilation,  # Equivalent to 'same' in TensorFlow, given a stride of 1.
            kernel_size=3,
            bias=False,  # No bias when using batch norm
        )
        # Apply Kaiming initialization to convolutional weights
        leaky_relu_negative_slope = 0.2
        nn.init.kaiming_normal_(self.conv.weight, a=leaky_relu_negative_slope)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.adaptive_batch_norm = AdaptiveBatchNorm1d(num_features=out_channels)

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        conv_t = self.conv(input_t)
        relu_t = self.leaky_relu(conv_t)
        return self.adaptive_batch_norm(relu_t)


class AdaptiveBatchNorm1d(nn.Module):
    """
    Apply adaptive batch normalization
    as defined in Fast Image Processing with Fully-Convolutional Networks
    """

    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.beta = nn.Parameter(torch.tensor([0.0]))

    def forward(self, input_t):
        norm_t = self.batch_norm(input_t)
        return self.alpha * input_t + self.beta * norm_t
