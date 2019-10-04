import torch
import torch.nn as nn

MIN_SAMPLES = 32767


class SceneNet(nn.Module):
    """
    Convolutional network used to classify acoustic scenes.
    """

    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.conv_1 = ConvLayer(in_channels=1, out_channels=32, stride=2)
        self.conv_2 = ConvLayer(in_channels=32, out_channels=32, stride=2)
        self.conv_3 = ConvLayer(in_channels=32, out_channels=32, stride=2)
        self.conv_4 = ConvLayer(in_channels=32, out_channels=32, stride=2)
        self.conv_5 = ConvLayer(in_channels=32, out_channels=32, stride=2)
        self.conv_6 = ConvLayer(in_channels=32, out_channels=64, stride=2)
        self.conv_7 = ConvLayer(in_channels=64, out_channels=64, stride=2)
        self.conv_8 = ConvLayer(in_channels=64, out_channels=64, stride=2)
        self.conv_9 = ConvLayer(in_channels=64, out_channels=64, stride=2)
        self.conv_10 = ConvLayer(in_channels=64, out_channels=128, stride=2)
        self.conv_11 = ConvLayer(in_channels=128, out_channels=128, stride=2)
        self.conv_12 = ConvLayer(in_channels=128, out_channels=128, stride=2)
        self.conv_13 = ConvLayer(in_channels=128, out_channels=128, stride=2)
        self.conv_14 = ConvLayer(in_channels=128, out_channels=128, stride=2)
        self.conv_15 = ConvLayer(in_channels=128, out_channels=128, stride=1)
        self.linear = nn.Linear(in_features=128, out_features=num_labels, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.LogSoftmax(dim=1)

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
        # torch.Size([1, 1, 32767])
        conv_acts = self.conv_1(input_t)
        # torch.Size([1, 32, 16384])
        conv_acts = self.conv_2(conv_acts)
        # torch.Size([1, 32, 8192])
        conv_acts = self.conv_3(conv_acts)
        # torch.Size([1, 32, 4096])
        conv_acts = self.conv_4(conv_acts)
        # torch.Size([1, 32, 2048])
        conv_acts = self.conv_5(conv_acts)
        # torch.Size([1, 32, 1024])
        conv_acts = self.conv_6(conv_acts)
        # torch.Size([1, 64, 512])
        conv_acts = self.conv_7(conv_acts)
        # torch.Size([1, 64, 256])
        conv_acts = self.conv_8(conv_acts)
        # torch.Size([1, 64, 128])
        conv_acts = self.conv_9(conv_acts)
        # torch.Size([1, 64, 64])
        conv_acts = self.conv_10(conv_acts)
        # torch.Size([1, 128, 32])
        conv_acts = self.conv_11(conv_acts)
        # torch.Size([1, 128, 16])
        conv_acts = self.conv_12(conv_acts)
        # torch.Size([1, 128, 8])
        conv_acts = self.conv_13(conv_acts)
        # torch.Size([1, 128, 4])
        conv_acts = self.conv_14(conv_acts)
        # torch.Size([1, 128, 2])
        conv_acts = self.conv_15(conv_acts)
        # torch.Size([1, 128, 2])

        # Perform average pooling over features to produce a standard 1D feature vector.
        pooled_acts = torch.mean(conv_acts, dim=2)
        # (batch_size, num_channels)
        # torch.Size([1, 128])

        # # Run linear layer over pooled features
        linear_acts = self.linear(pooled_acts)
        # Fight overfitting during training with dropout.
        linear_acts = self.dropout(linear_acts)
        # (batch_size, num_labels, 1)
        assert linear_acts.shape[0] == batch_size
        assert linear_acts.shape[1] == self.num_labels
        # torch.Size([1, 15])

        # # Run softmax over activations to produce the final probability distribution
        prediction_t = self.softmax(linear_acts)
        # torch.Size([1, 15])
        assert prediction_t.shape[1] == self.num_labels
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
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
        # Apply Kaiming initialization to convolutional weights
        nn.init.kaiming_normal_(self.conv.weight)
        # Adaptive norm parameters
        # self.adapt_alpha = nn.Parameter(torch.tensor([1.0]))
        # self.adapt_beta = nn.Parameter(torch.tensor([0.0]))

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        conv_t = self.conv(input_t)
        relu_t = self.lrelu(conv_t)
        norm_t = self.batch_norm(relu_t)
        # Apply adaptive normalization (Fast Image Processing with Fully-Convolutional Networks)
        # adapt_t = self.adapt_alpha * relu_t + self.adapt_beta * norm_t
        # return adapt_t
        return norm_t
