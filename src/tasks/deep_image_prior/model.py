"""
Skip net from Deep Image Prior: encoder-decoder with skip connections.
https://github.com/DmitryUlyanov/deep-image-prior
"""
import torch
from torch import nn


class SkipNet(nn.Module):
    def __init__(self, num_layers=5):
        super().__init__()
        # Build encoders
        self.encoders = nn.ModuleList()
        in_channels = 32
        for idx in range(num_layers):
            layer = get_encoder_layer(in_c=in_channels, out_c=128)
            self.encoders.append(layer)
            in_channels = 128

        # Build skips
        self.skips = nn.ModuleList()
        for idx in range(num_layers):
            layer = get_skip_layer(in_c=128, out_c=4)
            self.skips.append(layer)

        # Build decoders
        self.decoders = nn.ModuleList()
        for idx in range(num_layers):
            layer = get_decoder_layer(in_c=132, out_c=128)
            self.decoders.append(layer)

        self.out_conv = nn.Conv2d(128, 3, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_t):
        # (1, 32, 256, 256)
        skip_inputs = []
        acts = input_t
        for idx, encoder in enumerate(self.encoders):
            acts = encoder(acts)
            skip_inputs.append(acts)

        skip_outputs = []
        for idx, skip in enumerate(self.skips):
            skip_input = skip_inputs[idx]
            skip_output = skip(skip_input)
            skip_outputs.append(skip_output)

        # (1, 128, 64, 64)
        skip_outputs = list(reversed(skip_outputs))
        for idx, decoder in enumerate(self.decoders):
            skip_output = skip_outputs[idx]
            acts = torch.cat([skip_output, acts], dim=1)
            acts = decoder(acts)

        # (1, 128, 256, 256)
        acts = self.out_conv(acts)
        # (1, 3, 256, 256)
        return self.sigmoid(acts)


def get_decoder_layer(in_c, out_c):
    padder = nn.ReflectionPad2d(padding=1)
    layers = [
        nn.BatchNorm2d(num_features=in_c),
        padder,
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=0),
        nn.BatchNorm2d(num_features=out_c),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1, padding=0),
        nn.BatchNorm2d(num_features=out_c),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode="bilinear"),
    ]
    return nn.Sequential(*layers)


def get_skip_layer(in_c, out_c):
    layers = [
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=1, padding=0),
        nn.BatchNorm2d(num_features=out_c),
        nn.LeakyReLU(),
    ]
    return nn.Sequential(*layers)


def get_encoder_layer(in_c, out_c):
    padder = nn.ReflectionPad2d(padding=1)
    layers = [
        padder,
        nn.Conv2d(
            in_channels=in_c, out_channels=out_c, kernel_size=3, padding=0, stride=2
        ),
        nn.BatchNorm2d(num_features=out_c),
        nn.LeakyReLU(),
        padder,
        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=0),
        nn.BatchNorm2d(num_features=out_c),
        nn.LeakyReLU(),
    ]
    return nn.Sequential(*layers)
