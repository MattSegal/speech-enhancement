import torch
import torch.nn as nn


MIN_SAMPLES = 32767
CONV_LAYERS = [
    # in, out, stride
    [1, 32, 2],
    [32, 32, 2],
    # [32, 32, 2],
    # [32, 32, 2],
    # [32, 32, 2],
    [32, 64, 2],
    # [64, 64, 2],
    # [64, 64, 2],
    [64, 64, 2],
    [64, 128, 2],
    # [128, 128, 2],
    # [128, 128, 2],
    # [128, 128, 2],
    [128, 128, 2],
    [128, 128, 1],
]


class SceneNet(nn.Module):
    """
    Convolutional network used to classify acoustic scenes.
    """

    CHIME = "CHIME"
    TUT = "TUT"
    NUM_CHIME_LABELS = 8
    NUM_TUT_LABELS = 15
    dataset = TUT

    def __init__(self):
        super().__init__()
        conv_layers = [
            ConvLayer(in_channels, out_channels, stride=stride)
            for in_channels, out_channels, stride in CONV_LAYERS
        ]
        self.conv = nn.Sequential(*conv_layers)
        # Used for TUT dataset classfication
        self.softmax = nn.LogSoftmax(dim=1)
        self.tut_conv = nn.Conv1d(
            in_channels=128,
            out_channels=self.NUM_TUT_LABELS,
            kernel_size=1,
            bias=True,
        )
        # Used for CHiME datset classification
        self.sigmoid = nn.Sigmoid()
        self.chime_conv = nn.Conv1d(
            in_channels=128,
            out_channels=self.NUM_CHIME_LABELS,
            kernel_size=1,
            bias=True,
        )

    def forward(self, input_t):
        """
        Input has shape (batch_size, channels, audio,) 
        """
        batch_size = input_t.shape[0]
        # Convolutional filter expects a 3D input
        assert len(input_t.shape) == 3  # batch_size, channels, audio
        assert input_t.shape[1] == 1  # only 1 channel initially
        assert input_t.shape[2] >= MIN_SAMPLES  # Receptive field minimum

        # Pass input through a series of 1D covolutions
        # (batch_size, channels, length)
        # torch.Size([256, 1, 32767+])
        conv_acts = self.conv(input_t)
        # torch.Size([256, 128, 2+])

        # Perform average pooling over features to produce a standard 1D feature vector.
        pooled_acts = torch.mean(conv_acts, dim=2, keepdim=True)
        # (batch_size, num_channels)
        # torch.Size([256, 128, 1])

        if self.dataset == self.TUT:
            # Pool channels with 1x1 convolution
            final_conv_acts = self.tut_conv(pooled_acts)
            # (batch_size, num_labels, 1)
            # torch.Size([256, 15, 1])

            final_conv_acts = final_conv_acts.squeeze(dim=2)
            assert final_conv_acts.shape[0] == batch_size
            assert final_conv_acts.shape[1] == self.NUM_TUT_LABELS
            # torch.Size([256, 15])

            # # Run softmax over activations to produce the final probability distribution
            prediction_t = self.softmax(final_conv_acts)
            # torch.Size([256, 15])
            assert prediction_t.shape[1] == self.NUM_TUT_LABELS
            return prediction_t
        else:
            # Pool channels with 1x1 convolution
            final_conv_acts = self.chime_conv(pooled_acts)
            # (batch_size, num_labels, 1)
            # torch.Size([256, 8, 1])

            final_conv_acts = final_conv_acts.squeeze(dim=2)
            assert final_conv_acts.shape == (batch_size, self.NUM_CHIME_LABELS)
            # torch.Size([256, 8])

            # Run sigmoid over activation to get a probability
            prediction_t = self.sigmoid(final_conv_acts)
            # torch.Size([256, 8])
            assert prediction_t.shape == (batch_size, self.NUM_CHIME_LABELS)
            return prediction_t

    def set_tut_dataset(self):
        self.dataset = self.TUT

    def set_chime_dataset(self):
        self.dataset = self.CHIME

    @property
    def num_labels(self):
        if self.dataset == self.CHIME:
            return self.NUM_CHIME_LABELS
        else:
            return self.NUM_TUT_LABELS


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
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        # Apply Kaiming initialization to convolutional weights
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        conv_t = self.conv(input_t)
        relu_t = self.lrelu(conv_t)
        norm_t = self.batch_norm(relu_t)
        return norm_t
