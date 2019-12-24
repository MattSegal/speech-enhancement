import torch
from torch import nn
from torch.nn.utils import weight_norm


NUM_CHANNELS = 24  # Factor which determines the number of channels
NUM_LAYERS = 12


class WaveUNet(nn.Module):
    """
    Convolutional neural net for speech enhancement
    Proposed in Improved Speech Enhancement with the Wave-U-Net (https://arxiv.org/pdf/1811.11307.pdf),
    which builds upon this paper (https://arxiv.org/pdf/1806.03185.pdf)
    """

    def __init__(self, num_channels=NUM_CHANNELS, num_layers=NUM_LAYERS):
        super().__init__()
        self.skips_enabled = True
        # Construct encoders
        self.encoders = nn.ModuleList()
        layer = ConvLayer(1, num_channels, kernel=15)
        self.encoders.append(layer)
        for layer_idx in range(1, num_layers):
            in_channels = layer_idx * num_channels
            out_channels = (layer_idx + 1) * num_channels
            layer = ConvLayer(in_channels, out_channels, kernel=15)
            self.encoders.append(layer)

        self.middle = ConvLayer(12 * num_channels, 13 * num_channels, kernel=1)

        # Construct decoders
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.decoders = nn.ModuleList()
        for layer_idx in reversed(range(1, num_layers + 1)):
            in_channels = (2 * (layer_idx + 1) - 1) * num_channels
            out_channels = layer_idx * num_channels
            layer = ConvLayer(in_channels, out_channels, kernel=15)
            self.decoders.append(layer)

        # Extra dimension for input
        self.output = ConvLayer(num_channels + 1, 1, kernel=1, nonlinearity=nn.Tanh)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_encoder(self):
        for param in self.encoders.parameters():
            param.requires_grad = False

    def forward(self, input_t):
        batch_size = input_t.shape[0]
        # Encoding
        # (b, 1, 16384)
        input_t = input_t.view(batch_size, 1, -1)
        acts = input_t
        skip_connections = []
        for idx, encoder in enumerate(self.encoders):
            acts = encoder(acts)
            if self.skips_enabled or idx > NUM_LAYERS // 2 or True:
                # Store skip connections for the decoder phase
                skip_connections.append(acts)
            else:
                # Use zeros instead of skip connections
                zero_acts = torch.zeros_like(acts).detach()
                skip_connections.append(zero_acts)

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

        # (b, 24, 16384)
        acts = torch.cat((acts, input_t), dim=1)
        output_t = self.output(acts)
        # (batch, 1, 16384) (or 1, 3, 5, etc.)
        output_t = output_t.squeeze(dim=1)
        return output_t


class ConvLayer(nn.Module):
    """
    Single convolutional layer with nonlinear output
    """

    def __init__(self, in_channels, out_channels, kernel, nonlinearity=nn.PReLU):
        super().__init__()
        self.nonlinearity = nonlinearity()
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            padding=kernel // 2,  # Same padding
            bias=True,
        )
        self.conv = weight_norm(conv)
        # Apply Xavier initialization to convolutional weights
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        acts = self.conv(input_t)
        return self.nonlinearity(acts)
