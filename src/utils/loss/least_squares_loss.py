import torch


class LeastSquaresLoss:
    """
    Least squares loss for GAN training.
    """

    def __init__(self, disc_net):
        self.disc_net = disc_net

    def for_generator(self, real_audio, fake_audio):
        disc_fake = self.disc_net(fake_audio)
        return torch.mean((disc_fake - 1) ** 2)

    def for_discriminator(self, real_audio, fake_audio):
        disc_real = self.disc_net(real_audio)
        disc_fake = self.disc_net(fake_audio)
        return torch.mean((disc_real - 1) ** 2) + torch.mean(disc_fake ** 2)
