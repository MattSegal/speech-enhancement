import torch
from torch import nn
from torch.nn.utils import weight_norm

NUM_C = 24  # Factor which determines the number of channels
NUM_ENCODER_LAYERS = 12


class SpectralUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct encoders
        self.encoders = nn.ModuleList()
        self.encoder_res_stacks = nn.ModuleList()
        layer = ConvLayer(1, NUM_C, kernel=15)
        self.encoders.append(layer)
        for i in range(1, NUM_ENCODER_LAYERS):
            in_channels = i * NUM_C
            out_channels = (i + 1) * NUM_C
            layer = ConvLayer(in_channels, out_channels, kernel=15)
            self.encoders.append(layer)
            res_stack = ResidualDilationStack(num_channels=out_channels)
            self.encoder_res_stacks.append(res_stack)

        self.middle = ConvLayer(12 * NUM_C, 13 * NUM_C, kernel=15)

        # Construct decoders
        self.upsample = nn.Upsample(
            scale_factor=2, mode="linear", align_corners=True
        )
        self.decoders = nn.ModuleList()
        self.decoder_res_stacks = nn.ModuleList()
        for i in reversed(range(1, NUM_ENCODER_LAYERS + 1)):
            in_channels = (2 * (i + 1) - 1) * NUM_C
            out_channels = i * NUM_C
            layer = ConvLayer(in_channels, out_channels, kernel=5)
            self.decoders.append(layer)
            res_stack = ResidualDilationStack(num_channels=out_channels)
            self.decoder_res_stacks.append(res_stack)

        # Extra dimension for input
        self.output = ConvLayer(NUM_C + 1, 1, kernel=1, nonlinearity=nn.Tanh)

    def forward(self, input_t):
        # Encoding
        # (b, 1, 16384)
        acts = input_t
        skip_connections = []
        for idx, encoder in enumerate(self.encoders):
            acts = encoder(acts)
            if idx > 0:
                acts = self.encoder_res_stacks[idx - 1](acts)

            skip_connections.append(acts)
            # Decimate activations
            acts = acts[:, :, ::2]

        # (b, 288, 4)
        acts = self.middle(acts)
        # (b, 312, 4)

        # Decoding
        skip_connections = list(reversed(skip_connections))
        for idx, decoder in enumerate(self.decoders):
            # Upsample in the time direction by a factor of two, using interpolation
            acts = self.upsample(acts)
            # Concatenate upsampled input and skip connection from encoding stage.
            # Perform the concatenation in the feature map dimension.
            skip = skip_connections[idx]
            acts = torch.cat((acts, skip), dim=1)
            acts = decoder(acts)
            acts = self.decoder_res_stacks[idx](acts)

        # (b, 24, 16384)
        acts = torch.cat((acts, input_t), dim=1)
        output_t = self.output(acts)
        # (batch, 1, 16384) (or 1, 3, 5, etc.)
        return output_t


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
