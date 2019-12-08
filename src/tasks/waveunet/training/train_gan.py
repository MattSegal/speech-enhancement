"""
Train WaveUNet on the noisy VCTK dataset using MSE + GAN

Batch size of 32 uses approx 5GB of GPU memory.

Uses NoGAN training schedule
https://github.com/jantic/DeOldify#what-is-nogan
"""
import torch
import torch.nn as nn

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
    trainer.setup_wandb(
        WANDB_PROJECT,
        wandb_name,
        config={
            "Batch Size": batch_size,
            "Epochs": num_epochs,
            "Adam Betas": ADAM_BETAS,
            "Learning Rate": LEARNING_RATE,
            "Disc Learning Rate": DISC_LEARNING_RATE,
            "Disc Weight": DISC_WEIGHT,
            "Weight Decay": WEIGHT_DECAY,
            "Fine Tuning": False,
        },
    )

    # Construct generator network
    gen_net = trainer.load_net(WaveUNet)
    gen_optimizer = trainer.load_optimizer(
        gen_net,
        learning_rate=LEARNING_RATE,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    train_loader, test_loader = trainer.load_data_loaders(
        NoisySpeechDataset, batch_size, subsample
    )

    # Construct discriminator network
    disc_net = trainer.load_net(MelDiscriminatorNet)
    disc_loss = LeastSquaresLoss(disc_net)
    disc_optimizer = trainer.load_optimizer(
        disc_net,
        learning_rate=DISC_LEARNING_RATE,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    # First, train generator using MSE loss
    disc_net.freeze()
    gen_net.unfreeze()
    trainer.register_loss_fn(get_mse_loss)
    trainer.register_metric_fn(get_mse_metric, "Loss")
    trainer.input_shape = [2 ** 15]
    trainer.target_shape = [2 ** 15]
    trainer.output_shape = [2 ** 15]
    trainer.train(gen_net, num_epochs, gen_optimizer, train_loader, test_loader)

    # Next, train GAN using the output of the generator
    def get_disc_loss(_, fake_audio, real_audio):
        """
        We want to compare the inputs (real audio) with the generated outout (fake audio)
        """
        return disc_loss.for_discriminator(real_audio, fake_audio)

    def get_disc_metric(_, fake_audio, real_audio):
        loss_t = disc_loss.for_discriminator(real_audio, fake_audio)
        return loss_t.data.item()

    disc_net.unfreeze()
    gen_net.freeze()
    trainer.loss_fns = []
    trainer.metric_fns = []
    trainer.register_loss_fn(get_disc_loss)
    trainer.register_metric_fn(get_disc_metric, "Discriminator Loss")
    trainer.train(gen_net, num_epochs, disc_optimizer, train_loader, test_loader)

    # Finally, train the generator using the discriminator and MSE loss
    def get_gen_loss(_, fake_audio, real_audio):
        return disc_loss.for_generator(real_audio, fake_audio)

    def get_gen_metric(_, fake_audio, real_audio):
        loss_t = disc_loss.for_generator(real_audio, fake_audio)
        return loss_t.data.item()

    disc_net.freeze()
    gen_net.unfreeze()
    trainer.loss_fns = []
    trainer.metric_fns = []
    trainer.register_loss_fn(get_mse_loss)
    trainer.register_loss_fn(get_gen_loss, weight=DISC_WEIGHT)
    trainer.register_metric_fn(get_mse_metric, "Loss")
    trainer.register_metric_fn(get_gen_metric, "Generator Loss")
    trainer.train(gen_net, num_epochs, gen_optimizer, train_loader, test_loader)


def get_mse_loss(inputs, outputs, targets):
    return mse(outputs, targets)


def get_mse_metric(inputs, outputs, targets):
    mse_t = mse(outputs, targets)
    return mse_t.data.item()

