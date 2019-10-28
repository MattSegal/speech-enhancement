import torch


class RelativisticLoss:
    """
    Applies relativistic least squares loss as explained here
    https://ajolicoeur.wordpress.com/relativisticgan/ 

    I don't really get it, this is script kiddie territory ;)
    """

    def __init__(self, disc_net):
        self.disc_net = disc_net

    def for_generator(self, real_audio, fake_audio):
        disc_real = self.disc_net(real_audio)
        disc_fake = self.disc_net(fake_audio)
        return self._relativistic_least_square_loss_for_gen(disc_real, disc_fake)

    def for_discriminator(self, real_audio, fake_audio):
        disc_real = self.disc_net(real_audio)
        disc_fake = self.disc_net(fake_audio)
        return self._relativistic_least_square_loss_for_disc(disc_real, disc_fake)

    def _relativistic_least_square_loss_for_gen(self, disc_real, disc_fake):
        a = torch.mean((disc_real - torch.mean(disc_fake) + 1) ** 2)
        b = torch.mean((disc_fake - torch.mean(disc_real) - 1) ** 2)
        return a + b

    def _relativistic_least_square_loss_for_disc(self, disc_real, disc_fake):
        a = torch.mean((disc_real - torch.mean(disc_fake) - 1) ** 2)
        b = torch.mean((disc_fake - torch.mean(disc_real) + 1) ** 2)
        return a + b
