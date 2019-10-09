import torch
import torch.nn as nn

NUM_INNER_CONVS = 12
CHANNELS = 32

class SpeechDenoiseNet(nn.Module):
    """
    Convolutional network used to denoise human speech in audio.
    """

    def __init__(self):
        # 256, 1, 32768
        self.input_conv = ConvLayer(in_channels=1, out_channels=CHANNELS, dilation=1)
        self.inner_convs = nn.Sequential(*[
            ConvLayer(in_channels=CHANNELS, out_channels=CHANNELS, dilation=2**i)
            for i in range(1, NUM_INNER_CONVS + 1)    
        ])
        self.final_inner_conv = self.ConvLayer(in_channels=CHANNELS, out_channels=CHANNELS, dilation=1)
        # 256, 32, 32768
        self.output_conv = nn.Conv1d(
            in_channels=CHANNELS
            out_channels=1,
            kernel_size=1,
            bias=True,
        )
        # 256, 1, 32768


    def forward(self, input_t):
        """
        Input has shape (batch_size, 1, audio_length,) 
        Output has shape (batch_size, 1, audio_length,) 
        """
        batch_size = input_t.shape[0]
        assert input_t.shape[1] == 1
        audio_length = input_t.shape[2]
        acts = self.input_conv(input_t)
        assert input_t.shape == (batch_size, CHANNELS, audio_length)
        acts = self.inner_convs(acts)
        assert input_t.shape == (batch_size, CHANNELS, audio_length)
        acts = self.final_inner_conv(acts)
        assert input_t.shape == (batch_size, CHANNELS, audio_length)
        acts = self.output_conv(acts)
        assert input_t.shape == (batch_size, 1, audio_length)
        return acts


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
