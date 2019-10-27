import torch
import torch.nn as nn


class MultiScaleLoss:
    """
    Applies multi-scale loss function, as done in MelGAN
    Inspired by https://github.com/seungwonpark/melgan/blob/master/model/multiscale.py
    """

    def __init__(self, loss_fn, num_scales=2):
        self.loss_fn = loss_fn
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
        self.num_scales = num_scales

    def for_generator(self, real_audio, fake_audio):
        return self._get_loss(real_audio, fake_audio, self.loss_fn.for_generator)

    def for_discriminator(self, real_audio, fake_audio):
        return self._get_loss(real_audio, fake_audio, self.loss_fn.for_discriminator)

    def _get_loss(self, real_audio, fake_audio, loss_fn):
        loss = 0
        real = real_audio
        fake = fake_audio
        for _ in range(self.num_scales + 1):
            loss += loss_fn(real, fake)
            real = self.downsample(real)
            fake = self.downsample(fake)

        return loss / (self.num_scales + 1)
