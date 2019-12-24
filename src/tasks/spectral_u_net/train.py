"""
Train UNet on a spectrogram of the noisy VCTK dataset using MSE
"""
import torch.nn as nn

from src.datasets import NoisySpectralSpeechDataset as Dataset
from src.utils.trainer import Trainer

from .model import SpectralUNet

# Checkpointing
WANDB_PROJECT = "spectral-u-net"
CHECKPOINT_NAME = "spectral-u-net"

# Training hyperparams
MIN_LR = 2e-4
MAX_LR = 1e-3
ADAM_BETAS = (0.9, 0.99)
WEIGHT_DECAY = 1e-6

mse = nn.MSELoss()


def train(num_epochs, use_cuda, batch_size, wandb_name, subsample, checkpoint_epochs):
    trainer = Trainer(use_cuda, wandb_name)
    trainer.setup_checkpoints(CHECKPOINT_NAME, checkpoint_epochs)
    trainer.setup_wandb(
        WANDB_PROJECT,
        wandb_name,
        config={
            "Batch Size": batch_size,
            "Epochs": num_epochs,
            "Adam Betas": ADAM_BETAS,
            "Learning Rate": [MIN_LR, MAX_LR],
            "Weight Decay": WEIGHT_DECAY,
            "Fine Tuning": False,
        },
    )
    train_loader, test_loader = trainer.load_data_loaders(Dataset, batch_size, subsample)
    trainer.register_loss_fn(get_mse_loss)
    trainer.register_metric_fn(get_mse_metric, "Loss")
    trainer.input_shape = [1, 80, 256]
    trainer.target_shape = [1, 80, 256]
    trainer.output_shape = [1, 80, 256]
    net = trainer.load_net(SpectralUNet)
    optimizer = trainer.load_optimizer(
        net, learning_rate=MIN_LR, adam_betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
    )

    # One cycle learning rate
    steps_per_epoch = len(trainer.train_set) // batch_size
    trainer.use_one_cycle_lr_scheduler(optimizer, steps_per_epoch, num_epochs, MAX_LR)

    trainer.train(net, num_epochs, optimizer, train_loader, test_loader)


def get_mse_loss(inputs, outputs, targets):
    return mse(outputs, targets)


def get_mse_metric(inputs, outputs, targets):
    mse_t = mse(outputs, targets)
    return mse_t.data.item()
