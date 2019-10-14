import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint_sequential

"""
ADAM optimization algorithm, a learning rate of 0.0001, decay rates β1 = 0.9 and β2 = 0.999
a batch size of 16.

We specify an initial network layer size of 12
16 extra filters per layer are also specified,
with downsampling block filters of size 15 and upsampling block filters of size 5 like in [12].

We train for 2,000 iterations with mean squared error (MSE) over all source output samples in a batch as
loss and apply early stopping if there is no improvement on the validation set for 20 epochs.

We use a fixed validation set of 10 randomly selected tracks. 

Then, the best model is fine-tuned with the batch size doubled and 
the learning rate lowered to 0.00001, again until 20 epochs have passed without
improved validation loss.
"""


class WaveUNet(nn.Module):
    """
    Convolutional neural net for speech enhancement
    Proposed in Improved Speech Enhancement with the Wave-U-Net (https://arxiv.org/pdf/1811.11307.pdf),
    which was in turn inspired by this paper (https://arxiv.org/pdf/1806.03185.pdf)
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_t):
        """
        Input has shape (batch_size, 1, audio_length,) 
        Output has shape (batch_size, audio_length,) 
        """
        # (batch, 1, 16384)

        # start downsampling
        # repeat X times
        # conv1d
        # decimate channels
        # (batch, 4, 288) ?

        # end of downsampling segment
        # conv1d

        # repeat Y times
        # upsample
        # concat with sister layer
        # conv1d

        # concat input
        # conv1d with K output channels
