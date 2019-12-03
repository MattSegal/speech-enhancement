"""
Train WaveUNet on the noisy VCTK dataset using MSE + GAN

Batch size of ?? uses approx ??GB of GPU memory.

TODO: Add code from gan_old then implement NoGAN training schedule
https://github.com/jantic/DeOldify#what-is-nogan
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.datasets import NoisySpeechDataset
from src.utils.loss import LeastSquaresLoss
from src.utils.trainer import Trainer

from ..models.wave_u_net import WaveUNet
from ..models.mel_discriminator import MelDiscriminatorNet

# Checkpointing
WANDB_PROJECT = "wave-u-net"
CHECKPOINT_NAME = "wave-u-net"

# Training hyperparams
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-4
DISC_WEIGHT = 1e-1
DISC_LEARNING_RATE = 4 * LEARNING_RATE


mse = nn.MSELoss()


def train(num_epochs, use_cuda, batch_size, wandb_name, subsample, checkpoint_epochs):
    trainer = Trainer(num_epochs, wandb_name)
    trainer.setup_checkpoints(CHECKPOINT_NAME, checkpoint_epochs)
    trainer.setup_wandb(WANDB_PROJECT, wandb_name)

    # Construct generator network
    gen_net = trainer.load_net(WaveUNet)
    gen_optimizer = trainer.load_optimizer(
        gen_net,
        learning_rate=LEARNING_RATE,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    gen_train_loader, gen_test_loader = trainer.load_data_loaders(
        NoisySpeechDataset, batch_size, subsample
    )

    # Construct discriminator network
    disc_net = trainer.load_net(MelDiscriminatorNet)
    disc_optimizer = trainer.load_optimizer(
        disc_net,
        learning_rate=DISC_LEARNING_RATE,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    disc_train_dataset = GeneratorDataset(gen_net, trainer.train_set)
    disc_test_dataset = GeneratorDataset(gen_net, trainer.test_set)
    disc_train_loader = trainer.load_data_loader(disc_train_dataset, batch_size)
    disc_test_loader = trainer.load_data_loader(disc_test_dataset, batch_size)
    disc_loss = LeastSquaresLoss(disc_net)

    # First, train generator using MSE loss
    trainer.register_loss_fn(get_mse_loss)
    trainer.register_metric_fn(get_mse_metric, "Loss")
    trainer.input_shape = [2 ** 15]
    trainer.target_shape = [2 ** 15]
    trainer.output_shape = [2 ** 15]
    trainer.train(gen_net, num_epochs, gen_optimizer, gen_train_loader, gen_test_loader)

    # Next, train GAN using the output of the generator
    def get_disc_loss(dunno1, dunno2, dunno3):
        return disc_loss.for_discriminator(dunno1, dunno2)

    def get_disc_metric(dunno1, dunno2, dunno3):
        loss_t = dic_loss.for_discriminator
        return loss_t.data.item()

    trainer.loss_fns = []
    trainer.metric_fns = []
    trainer.register_loss_fn(get_disc_loss)
    trainer.register_metric_fn(get_disc_metric, "Discriminator Loss")
    trainer.input_shape = [2 ** 15]
    trainer.target_shape = [2 ** 15]
    trainer.train(
        disc_net, num_epochs, disc_optimizer, disc_train_loader, disc_test_loader
    )

    # Finally, train the generator using the discriminator and MSE loss
    def get_gen_loss(dunno1, dunno2, dunno3):
        return disc_loss.for_generator(dunno1, dunno2)

    def get_gen_metric(dunno1, dunno2, dunno3):
        loss_t = dic_loss.for_generator
        return loss_t.data.item()

    trainer.loss_fns = []
    trainer.metric_fns = []
    trainer.register_loss_fn(get_mse_loss)
    trainer.register_loss_fn(get_gen_loss)
    trainer.register_metric_fn(get_mse_metric, "Loss")
    trainer.register_metric_fn(get_gen_metric, "Discriminator Loss")
    trainer.input_shape = [2 ** 15]
    trainer.target_shape = [2 ** 15]
    trainer.train(
        disc_net, num_epochs, disc_optimizer, disc_train_loader, disc_test_loader
    )


def get_mse_loss(inputs, outputs, targets):
    return mse(outputs, targets)


def get_mse_metric(inputs, outputs, targets):
    mse_t = mse(outputs, targets)
    return mse_t.data.item()


class GeneratorDataset(Dataset):
    def __init__(self, gen_net, dataset):
        self.gen_net = gen_net
        self.dataset = dataset

    def __getitem__(self, idx):
        """
        Get item by integer index,
        """
        noisy, real_clean = self.dataset[idx]
        with torch.no_grad():
            fake_clean = self.gen_net(noisy)

        return real_clean, fake_clean
