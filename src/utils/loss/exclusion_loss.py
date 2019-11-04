"""
Exclusion loss from Single Image Reflection Removal with Perceptual Losses
https://github.com/ceciliavision/perceptual-reflection-removal

A loss that emphasizes independence of the layers to be separated in the gradient domain.
Our key observation is that the edges of the transmission and the reflection layers are unlikely to overlap:
an edge in the source should be caused by either one layer or the other, but not both.
"""
import torch
from torch.nn.functional import sigmoid


def exclusion_loss(inputs, outputs, targets):
    audio_1 = outputs
    audio_2 = inputs.squeeze(dim=1) - targets
    grad_1 = get_gradient(audio_1)
    grad_2 = get_gradient(audio_2)
    alpha = 2 * torch.mean(torch.abs(grad_1)) / torch.mean(torch.abs(grad_2))
    grad_1_sig = (sigmoid(grad_1) * 2) - 1
    grad_2_sig = (sigmoid(grad_2 * alpha) * 2) - 1
    loss = torch.mean(grad_1_sig ** 2 * grad_2_sig ** 2, dim=1) ** 0.25
    return sum(loss)


def get_gradient(audio):
    """
    Get the gradient of an audio signal
    """
    return audio[:, 1:] - audio[:, :-1]
