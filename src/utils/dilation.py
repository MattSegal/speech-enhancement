"""
Repliactes "dilation" operation used in Deep Feature Loss paper.
Not really sure what this is doing, doesn't seem like convolutional dilation
as I understand it. 
"""


def signal_to_dilated(signal, dilation, n_channels):
    """
    Convert signal into "dilated" form
        signal shape [batch_size, n_channels, width, height]
        return shape [batch_size, ???, width, ???]

    """
    dilated = signal.reshape([signal.shape[0], -1, dilation, n_channels])
    return dilated.permute(0, 2, 1, 3)


def dilated_to_signal(dilated, n_channels):
    """
    Convert "dilated" form into something?
        dilated shape [batch_size, ???, width, ???]
        return shape  [batch_size, ???, width, ???]
    """
    shape = dilated.shape
    signal = dilated.permute(0, 2, 1, 3)
    signal = signal.reshape([shape[0], 1, -1, n_channels])
    return signal[:, :, : shape[1] * shape[2], :]

