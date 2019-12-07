import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# epoch |   batch | GPU memory  |   checkpoints
# 15s       32      2GB             0
# 15s       64      4.5GB           0
# 13s       128     4.7GB           0
# X         256     OOM             0
# X         128     4GB             conv layers

NUM_LABELS = 15
MIN_SAMPLES = 32767
NUM_SAMPLE_LAYERS = 6
CONV_LAYERS = [
    # in, out, stride
    [1, 32, 2],
    [32, 32, 2],
    [32, 32, 2],
    [32, 32, 2],
    [32, 32, 2],
    [32, 64, 2],
    [64, 64, 2],
    [64, 64, 2],
    [64, 64, 2],
    [64, 128, 2],
    [128, 128, 2],
    [128, 128, 2],
    [128, 128, 2],
    [128, 128, 2],
    [128, 128, 1],
]


class SceneNet(nn.Module):
    """
    Convolutional network used to classify acoustic scenes.
    """

    feature_mode = False

    def __init__(self):
        super().__init__()
        # Initializer feature layers, used for the denoiser loss function.
        self.feature_layers = []

        # Create internal convolution layers, for feature extraction.
        layer_idx = 0
        self.conv_layers = []
        for in_channels, out_channels, stride in CONV_LAYERS:
            layer_idx += 1
            layer_name = f"conv_{layer_idx}"
            conv_layer = ConvLayer(in_channels, out_channels, stride=stride)
            self.conv_layers.append(conv_layer)
            setattr(self, layer_name, conv_layer)

        self.final_conv = nn.Conv1d(
            in_channels=128, out_channels=NUM_LABELS, kernel_size=1, bias=True,
        )
        self.softmax = nn.LogSoftmax(dim=1)

    def set_feature_mode(self):
        self.feature_mode = True
        for param in self.parameters():
            param.requires_grad = False

        for layer in self.conv_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_t):
        """
        Input has shape (batch_size, channels, audio,) 
        """
        # Reset feature layers
        self.feature_layers = []

        # Convolutional filter expects a 3D input
        batch_size = input_t.shape[0]
        input_t = input_t.view(batch_size, 1, -1)
        assert len(input_t.shape) == 3  # batch_size, channels, audio
        assert input_t.shape[1] == 1  # only 1 channel initially
        assert input_t.shape[2] >= MIN_SAMPLES  # Receptive field minimum

        # Pass input through a series of 1D covolutions
        # (batch_size, channels, length)
        # torch.Size([256, 1, 32767+])
        acts = input_t
        for idx, conv_layer in enumerate(self.conv_layers):
            acts, conv_acts = checkpoint(conv_layer, acts)
            if idx < NUM_SAMPLE_LAYERS and self.feature_mode:
                # Store feature layers for feature loss.
                self.feature_layers.append(conv_acts)
            elif self.feature_mode:
                # Bail early because we don't care about the output.
                return

        # torch.Size([256, 128, 2+])
        # Perform average pooling over features to produce a standard 1D feature vector.
        pooled_acts = torch.mean(conv_acts, dim=2, keepdim=True)
        # (batch_size, num_channels)
        # torch.Size([256, 128, 1])

        # Pool channels with 1x1 convolution
        final_conv_acts = self.final_conv(pooled_acts)
        # (batch_size, num_labels, 1)
        # torch.Size([256, 15, 1])

        final_conv_acts = final_conv_acts.squeeze(dim=2)
        assert final_conv_acts.shape[0] == batch_size
        assert final_conv_acts.shape[1] == NUM_LABELS
        # torch.Size([256, 15])

        # # Run softmax over activations to produce the final probability distribution
        prediction_t = self.softmax(final_conv_acts)
        # torch.Size([256, 15])
        assert prediction_t.shape[1] == NUM_LABELS
        return prediction_t


class ConvLayer(nn.Module):
    """
    Single convolutional unit for the acoustic classifier network.
        Input tensor: (batch_size, in_channels, length)
        Output tensor: (batch_size, out_channels, length)
    """

    def __init__(self, in_channels, out_channels, stride):
        """
        Setup the layer.
            in_channels: number of input channels to be convoluted
            out_channels: number of output channels to be produced

        """
        super().__init__()
        # Configure padding so the input and output sizes are the same.

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=3,
            padding=1,
            bias=False,  # No bias when using batch norm
        )
        negative_slope = 0.2
        self.lrelu = nn.LeakyReLU(negative_slope=negative_slope)
        self.adaptive_batch_norm = AdaptiveBatchNorm1d(num_features=out_channels)
        # Apply Kaiming initialization to convolutional weights
        nn.init.kaiming_normal_(self.conv.weight, a=negative_slope)

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        conv_t = self.conv(input_t)
        relu_t = self.lrelu(conv_t)
        norm_t = self.adaptive_batch_norm(relu_t)
        return norm_t, conv_t


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
