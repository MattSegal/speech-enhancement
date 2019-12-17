"""
Train WaveUNet on the noisy VCTK dataset using MSE

Batch size of 32 uses approx 6GB of GPU memory.

From "Improved Speech Enhancement with the Wave-U-Net"

- Trained with MSE for 2000 epocs on 1e-4 LR
- Early stopping when no improvements on validation set for 20 epochs
- Fine tuning done with 1e-1 x LR and double batch size

"""
import torch.nn as nn

from src.datasets import NoisySpeechDataset as Dataset
from src.utils.trainer import Trainer

from ..models.wave_u_net import WaveUNet

# Checkpointing
WANDB_PROJECT = "wave-u-net"
CHECKPOINT_NAME = "wave-u-net"

# Training hyperparams
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-4

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
            "Learning Rate": LEARNING_RATE,
            "Weight Decay": WEIGHT_DECAY,
            "Fine Tuning": True,
        },
    )
    train_loader, test_loader = trainer.load_data_loaders(Dataset, batch_size, subsample)
    trainer.register_loss_fn(get_mse_loss)
    trainer.register_metric_fn(get_mse_metric, "Loss")
    trainer.input_shape = [2 ** 15]
    trainer.target_shape = [2 ** 15]
    trainer.output_shape = [2 ** 15]
    net = trainer.load_net(WaveUNet)
    optimizer = trainer.load_optimizer(
        net, learning_rate=LEARNING_RATE, adam_betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
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
