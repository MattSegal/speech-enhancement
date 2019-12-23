import torch
from torch import nn
from torch.nn.utils import weight_norm

from src.utils import spectral

NUM_ENCODER_LAYERS = 8
NUM_CHAN = 24  # Factor which determines the number of channels
NUM_INPUT_CHAN = 1


class SpectralUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct encoders
        self.encoders = nn.ModuleList()
        layer = ConvLayer(NUM_INPUT_CHAN, NUM_CHAN, kernel=[3, 5])
        self.encoders.append(layer)
        for i in range(1, NUM_ENCODER_LAYERS):
            in_channels = i * NUM_CHAN
            out_channels = (i + 1) * NUM_CHAN
            layer = ConvLayer(in_channels, out_channels, kernel=[3, 5])
            self.encoders.append(layer)

        self.middle = ConvLayer(
            NUM_ENCODER_LAYERS * NUM_CHAN,
            (NUM_ENCODER_LAYERS + 1) * NUM_CHAN,
            kernel=[1, 1],
        )

        # Construct decoders
        self.upsample_both = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.upsample_temporal = nn.Upsample(
            scale_factor=(1, 2), mode="bilinear", align_corners=True
        )
        self.decoders = nn.ModuleList()
        for i in reversed(range(1, NUM_ENCODER_LAYERS + 1)):
            in_channels = (2 * (i + 1) - 1) * NUM_CHAN
            out_channels = i * NUM_CHAN
            layer = ConvLayer(in_channels, out_channels, kernel=[3, 5])
            self.decoders.append(layer)

        self.final_conv = ConvLayer(
            NUM_CHAN + NUM_INPUT_CHAN, NUM_INPUT_CHAN, kernel=[1, 1], nonlinearity=None
        )

    def forward(self, input_t):
        # Encoding
        acts = input_t
        skip_connections = []
        for idx, encoder in enumerate(self.encoders):
            acts = encoder(acts)
            skip_connections.append(acts)
            # Decimate temporal activations
            acts = acts[:, :, :, ::2]
            if idx % 2 == 0:
                # Decimate spectral activations
                acts = acts[:, :, ::2, :]

        # Bridge between encoder and decoder
        acts = self.middle(acts)

        # Decoding
        skip_connections = list(reversed(skip_connections))
        for idx, decoder in enumerate(self.decoders):
            if idx % 2 == 1:
                # Upsample temporal + spectral
                acts = self.upsample_both(acts)
            else:
                # Upsample temporal only
                acts = self.upsample_temporal(acts)

            # Concatenate upsampled input and skip connection from encoding stage.
            # Perform the concatenation in the feature map dimension.
            skip = skip_connections[idx]
            acts = torch.cat((acts, skip), dim=1)
            acts = decoder(acts)

        # Concatenate decoded acts with input
        acts = torch.cat((acts, input_t), dim=1)
        output_t = self.final_conv(acts)
        return output_t


class ConvLayer(nn.Module):
    """
    Single convolutional layer with nonlinear output
    """

    def __init__(self, in_channels, out_channels, kernel, nonlinearity=nn.PReLU):
        super().__init__()
        self.nonlinearity = nonlinearity() if nonlinearity else None
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            padding=[
                kernel[0] // 2,
                kernel[1] // 2,
            ],  # Same padding in temporal direction
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
        if self.nonlinearity:
            return self.nonlinearity(acts)
        else:
            return acts
