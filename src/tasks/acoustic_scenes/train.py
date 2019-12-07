"""
Train SceneNet on the noisy TUT acoustic scenes dataset
to do acoustic scene classification.

Batch size of 128 works well, ~200 epochs required.
"""
import torch.nn as nn

from src.utils.trainer import Trainer
from src.datasets import SceneDataset as Dataset

from .model import SceneNet

# Checkpointing
WANDB_PROJECT = "chime-scene-net"
CHECKPOINT_NAME = "scene-net"

# Training hyperparams
import random


def from_exp(val):
    return 10 ** (-10 * val)


def to_exp(log_val):
    return math.log(val, 10) / -10


lr_exp = random.uniform(0.3, 0.6)
wd_exp = random.uniform(lr_exp, 0.8)

LEARNING_RATE = from_exp(lr_exp)
WEIGHT_DECAY = from_exp(wd_exp)
ADAM_BETAS = (random.triangular(0.4, 0.95, 0.8), random.triangular(0.8, 0.999, 0.95))
BATCH_SIZE = random.choice([2 ** n for n in range(3, 8)])


cross_entropy_loss = nn.CrossEntropyLoss()


def train(num_epochs, use_cuda, batch_size, wandb_name, subsample, checkpoint_epochs):
    batch_size = BATCH_SIZE
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
            "Fine Tuning": False,
        },
    )
    train_loader, test_loader = trainer.load_data_loaders(Dataset, batch_size, subsample)
    trainer.register_loss_fn(get_ce_loss)
    trainer.register_metric_fn(get_ce_metric, "Loss")
    trainer.input_shape = [32767]
    net = trainer.load_net(SceneNet)
    optimizer = trainer.load_optimizer(
        net, learning_rate=LEARNING_RATE, adam_betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
    )
    trainer.train(net, num_epochs, optimizer, train_loader, test_loader)

    # Do a fine tuning run with 1/10th learning rate for 1/3rd epochs.
    optimizer = trainer.load_optimizer(
        net, learning_rate=LEARNING_RATE / 10, adam_betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY / 10
    )
    num_epochs = num_epochs // 3
    trainer.train(net, num_epochs, optimizer, train_loader, test_loader)


def get_ce_loss(inputs, outputs, targets):
    return cross_entropy_loss(outputs, targets)


def get_ce_metric(inputs, outputs, targets):
    ce_t = cross_entropy_loss(outputs, targets)
    return ce_t.data.item()
