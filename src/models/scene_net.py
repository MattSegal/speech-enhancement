import torch
import torch.nn as nn


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
        self.linear = nn.Conv1d(
            in_channels=128,
            out_channels=num_labels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_t):
        """
        Input has shape (batch_size, channels, audio,) 
        """
        batch_size = input_t.shape[0]
        # Convolutional filter expects a 3D input
        assert len(input_t.shape) == 3  # batch_size, channels, audio
        assert input_t.shape[1] == 1  # only 1 channel initially

        # Pass input through a series of 1D covolutions
        # (batch_size, channels, length)
        # torch.Size([1, 1, 480000])
        conv_acts = self.conv_1(input_t)
        # torch.Size([1, 32, 239999])
        conv_acts = self.conv_2(conv_acts)
        # torch.Size([1, 32, 119999])
        conv_acts = self.conv_3(conv_acts)
        # torch.Size([1, 32, 59999])
        conv_acts = self.conv_4(conv_acts)
        # torch.Size([1, 32, 29999])
        conv_acts = self.conv_5(conv_acts)
        # torch.Size([1, 32, 14999])
        conv_acts = self.conv_6(conv_acts)
        # torch.Size([1, 64, 7499])
        conv_acts = self.conv_7(conv_acts)
        # torch.Size([1, 64, 3749])
        conv_acts = self.conv_8(conv_acts)
        # torch.Size([1, 64, 1874])
        conv_acts = self.conv_9(conv_acts)
        # torch.Size([1, 64, 936])
        conv_acts = self.conv_10(conv_acts)
        # torch.Size([1, 128, 467])
        conv_acts = self.conv_11(conv_acts)
        # torch.Size([1, 128, 233])
        conv_acts = self.conv_12(conv_acts)
        # torch.Size([1, 128, 116])
        conv_acts = self.conv_13(conv_acts)
        # torch.Size([1, 128, 57])
        conv_acts = self.conv_14(conv_acts)
        # torch.Size([1, 128, 28])
        conv_acts = self.conv_15(conv_acts)
        # torch.Size([1, 128, 26])

        # Perform average pooling over features to produce a standard 1D feature vector.
        pooled_acts = torch.mean(conv_acts, dim=2, keepdim=True)
        # (batch_size, channels)
        # torch.Size([1, 128])

        # # Run 1x1 convolutional filter over pooled features
        linear_acts = self.linear(pooled_acts)
        # (batch_size, num_labels, 1)
        # torch.Size([1, 15, 1])

        # # Run softmax over activations to produce the final probability distribution
        prediciton_acts = linear_acts.view(batch_size, self.num_labels)
        assert prediciton_acts.shape[1] == self.num_labels
        # torch.Size([1, 15])
        prediction_t = self.softmax(prediciton_acts)
        # torch.Size([1, 15])
        assert prediction_t.shape[1] == self.num_labels
        return prediction_t


class ConvLayer(nn.Module):
    """
    Single convolutional unit for the acoustic classifier network.
        Input tensor: (batch_size, in_channels, height, width)
        Output tensor: (batch_size, out_channels, height, width)

    TODO - add Xavier or Kaiming initialization
    https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

    """

    def __init__(self, in_channels, out_channels, stride):
        """
        Setup the layer.
            in_channels: number of input channels to be convoluted
            out_channels: number of output channels to be produced

        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            kernel_size=3,
            padding=0,  # Input size should be the same as output size?
            # FIXME - no padding - do not understand why it is used
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        acts_t = self.conv(input_t)
        acts_t = self.lrelu(acts_t)
        acts_t = self.batch_norm(acts_t)
        return acts_t
