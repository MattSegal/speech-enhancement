import torch.nn as nn

# Channel count for each layer's output
LAYER_CHANNELS = [32]
# , 32, 32, 32, 32, 64, 64, 64, 64, 64, 128, 128, 128, 128]


class SceneNet(nn.Module):
    """
    Convolutional network used to classify acoustic scenes.
    """

    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        # Build the sequence of convolutional layers
        conv_layers = []
        in_channels = 1
        for out_channels in LAYER_CHANNELS[:-1]:
            layer = ConvLayer(
                in_channels=in_channels, out_channels=out_channels, stride=[1, 2]
            )
            in_channels = out_channels
            conv_layers.append(layer)

        final_conv_layer = ConvLayer(
            in_channels=in_channels, out_channels=LAYER_CHANNELS[-1], stride=1
        )
        conv_layers.append(final_conv_layer)

        self.convolutions = nn.Sequential(*conv_layers)

        # self.average_pool = nn.AvgPool2d(
        #     # ????
        # )
        # self.linear = nn.Conv2d(
        #     in_channels=1, # is this right?
        #     out_channels=num_labels, # is this right?
        #     kernel_size=[1, 1],
        #     stride=1,
        #     padding=0,.
        # )
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, input_t):
        """
        Input has shape (signal_length,) 
        """
        # Pass activations through a series of covolutions
        conv_acts = self.convolutions(input_t)
        return conv_acts
        # # Perform average pooling on feature maps
        # pooled_acts = self.average_pool(conv_acts_t)
        # # Run 1x1 convolutional filter over pooled features
        # linear_acts = self.linear(pooled_acts)
        # # Run softmax over activations to produce the final probability distribution
        # prediction_t = self.softmax(linear_acts)
        # assert prediction_t.shape == (num_labels,)
        # return prediction_t


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
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=[1, 3],
            stride=[1, 2],
            padding=0,  # Input size should be the same as output size?
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input_t):
        """
        Compute output tensor from input tensor
        """
        acts_t = self.conv(input_t)
        acts_t = self.lrelu(acts_t)
        acts_t = self.batch_norm(acts_t)
        return acts_t
