"""
Train SpectralSceneNet on the noisy TUT acoustic scenes dataset
to do acoustic scene classification.

Batch size of 64-128 works well.
"""
import torch.nn as nn

from src.utils.trainer import Trainer
from src.datasets import SpectralSceneDataset as Dataset

from .model import SpectralSceneNet

# Checkpointing
WANDB_PROJECT = "spectral-scene-net"
CHECKPOINT_NAME = "spectral-scene-net"


MIN_LR = 1e-4
MAX_LR = 2e-4
ADAM_BETAS = (0.9, 0.99)
WEIGHT_DECAY = 1e-3


cross_entropy_loss = nn.CrossEntropyLoss()


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
    trainer.register_loss_fn(get_ce_loss)
    trainer.register_metric_fn(get_ce_metric, "Loss")
    trainer.register_metric_fn(get_accuracy_metric, "Accuracy")
    trainer.input_shape = [1, 80, 256]
    trainer.output_shape = [15]
    net = trainer.load_net(SpectralSceneNet)
    optimizer = trainer.load_optimizer(
        net, learning_rate=MIN_LR, adam_betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
    )

    # One cycle learning rate
    steps_per_epoch = len(trainer.train_set) // batch_size
    trainer.use_one_cycle_lr_scheduler(optimizer, steps_per_epoch, num_epochs, MAX_LR)

    trainer.train(net, num_epochs, optimizer, train_loader, test_loader)


def get_ce_loss(inputs, outputs, targets):
    return cross_entropy_loss(outputs, targets)


def get_ce_metric(inputs, outputs, targets):
    ce_t = cross_entropy_loss(outputs, targets)
    return ce_t.data.item()


def get_accuracy_metric(inputs, outputs, targets):
    predictions = outputs.argmax(dim=1)
    num_correct = (predictions == targets).sum().data.item()
    num_total = len(targets)
    return num_correct / num_total
