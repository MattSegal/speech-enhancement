"""
Train WaveUNet on the noisy VCTK dataset using feature loss + MSE

FIXME: ADD THIS BACK
"""
import torch.nn as nn


from src.datasets import NoisySpeechDataset as Dataset
from src.utils.loss import AudioFeatureLoss
from ..models.wave_u_net import WaveUNet

# Checkpointing
WANDB_PROJECT = "wave-u-net"
CHECKPOINT_NAME = "wave-u-net"

# Training hyperparams
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-6

from src.utils.trainer import Trainer

mse = nn.MSELoss()


def train(num_epochs, use_cuda, batch_size, wandb_name, subsample, checkpoint_epochs):
    trainer = Trainer(num_epochs, wandb_name)
    trainer.setup_checkpoints(CHECKPOINT_NAME, checkpoint_epochs)
    trainer.setup_wandb(WANDB_PROJECT, wandb_name)
    noise_data = NoisyScenesDataset(subsample=subsample)
    train_loader, test_loader = trainer.load_data_loaders(
        Dataset, batch_size, subsample, noise_data=noise_data
    )
    trainer.register_loss_fn(get_mse_loss)
    trainer.register_metric_fn(get_mse_metric, "Loss")
    trainer.input_shape = [2 ** 15]
    trainer.target_shape = [2 ** 15]
    trainer.output_shape = [2 ** 15]
    net = trainer.load_net(WaveUNet)
    optimizer = trainer.load_optimizer(
        net,
        learning_rate=LEARNING_RATE,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    trainer.train(net, num_epochs, optimizer, train_loader, test_loader)
    # Do a fine tuning run with 1/10th learning rate for 1/3rd epochs.
    optimizer = trainer.load_optimizer(
        net,
        learning_rate=LEARNING_RATE / 10,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY / 10,
    )
    num_epochs = num_epochs // 3
    trainer.train(net, num_epochs, optimizer, train_loader, test_loader)


def get_mse_loss(inputs, outputs, targets):
    return mse(outputs, targets)


def get_mse_metric(inputs, outputs, targets):
    mse_t = mse(outputs, targets)
    return mse_t.data.item()
