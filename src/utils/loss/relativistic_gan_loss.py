import torch
from torch import nn


class RelativisticAverageStandardGANLoss:
    """
    Applies Relativistic average Standard GAN (RaSGAN) discriminator
     as explained here: https://ajolicoeur.wordpress.com/relativisticgan/ 

    I don't really get it, this is script kiddie territory ;)
    """

    def __init__(self, disc_net):
        self.disc_net = disc_net
        self.loss_fn = nn.BCEWithLogitsLoss()

    def for_generator(self, real_audio, fake_audio):
        batch_size = real_audio.shape[0]
        zeros = torch.zeros([batch_size, 1], dtype=torch.float32).cuda()
        ones = torch.ones([batch_size, 1], dtype=torch.float32).cuda()
        disc_real = self.disc_net(real_audio)
        disc_fake = self.disc_net(fake_audio)
        mean_disc_fake = torch.mean(disc_fake)
        mean_disc_real = torch.mean(disc_real)
        a = self.loss_fn(disc_real - mean_disc_fake, zeros)
        b = self.loss_fn(disc_fake - mean_disc_real, ones)
        return a + b

    def for_discriminator(self, real_audio, fake_audio):
        batch_size = real_audio.shape[0]
        zeros = torch.zeros([batch_size, 1], dtype=torch.float32).cuda()
        ones = torch.ones([batch_size, 1], dtype=torch.float32).cuda()
        disc_real = self.disc_net(real_audio)
        disc_fake = self.disc_net(fake_audio)
        mean_disc_fake = torch.mean(disc_fake)
        mean_disc_real = torch.mean(disc_real)
        a = self.loss_fn(disc_real - mean_disc_fake, ones)
        b = self.loss_fn(disc_fake - mean_disc_real, zeros)
        return a + b
