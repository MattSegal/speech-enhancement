import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.checkpoint import checkpoint

NUM_LABELS = 15
CONV_LAYERS = [
    # in, out, pool
    # 80x256
    [1, 64, False],
    [64, 64, True],
    # 40x128
    [64, 128, False],
    [128, 128, True],
    # 20x64
    [128, 256, False],
    [256, 256, False],
    [256, 256, True],
    # 10x32
    [256, 512, False],
    [512, 512, False],
    [512, 512, True],
    # 5x16
]


class SpectralSceneNet(nn.Module):
    """
    Convolutional network used to classify acoustic scenes.
    Loosely based on VGG architecture.
    """

    num_feature_layers = 0

    def __init__(self):
        super().__init__()
        # Initializer feature layers, used for the denoiser loss function.
        self.feature_layers = []
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convs = nn.ModuleList()
        for in_c, out_c, is_pool in CONV_LAYERS:
            layer = ConvLayer(in_c, out_c, 3)
            self.convs.append(layer)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 5 * 16, 2046),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2046, 2046),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2046, NUM_LABELS),
            nn.LogSoftmax(dim=1),
        )

    def set_feature_mode(self, num_layers):
        self.num_feature_layers = num_layers
        for param in self.parameters():
            param.requires_grad = False

        return self.eval()

    def forward(self, input_t):
        """
        Input has shape (batch_size, channels, audio,) 
        """
        # Reset feature layers
        self.feature_layers = []

        # Convolutional filter expects a 4D input
        acts = input_t.view(-1, 1, 80, 256)

        # Run convolutional encoder
        for idx, (_, _, is_pool) in enumerate(CONV_LAYERS):
            acts, conv_t = self.convs[idx](acts)
            if idx < self.num_feature_layers:
                # Store feature layers for feature loss.
                self.feature_layers.append(conv_t)
            elif self.num_feature_layers and idx >= self.num_feature_layers:
                # Bail early because we don't care about the output.
                return

            if is_pool:
                acts = self.max_pool(acts)

        # Get predictions.
        acts = torch.flatten(acts, 1)
        pred_t = self.classifier(acts)
        return pred_t


class ConvLayer(nn.Module):
    """
    Single convolutional unit for the acoustic classifier network.
        Input tensor: (batch_size, in_channels, length)
        Output tensor: (batch_size, out_channels, length)
    """

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            padding=kernel // 2,
            bias=True,
        )
        self.conv = weight_norm(conv)
        nn.init.xavier_uniform_(self.conv.weight)
        self.prelu = nn.PReLU()

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        conv_t = self.conv(input_t)
        prelu_t = self.prelu(conv_t)
        return prelu_t, conv_t
