import random

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from ..datasets import NoisySpeechDataset
from ..models.wave_u_net import WaveUNet
from ..models.mel_discriminator import MelDiscriminatorNet
from ..models.scene_net import SceneNet
from ..utils.trackers import MovingAverage
from ..utils import checkpoint
from ..utils.log import log_training_info

# Checkpointing
LOSS_NET_CHECKPOINT = "saved/scene-net-long-train.ckpt"
CHECKPOINT_NAME = "wave-u-net"
WANDB_PROJECT = "wave-u-net"

# Training hyperparams
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-4


def train(num_epochs, use_cuda, wandb_name, subsample, checkpoint_epochs):
    use_wandb = bool(wandb_name)
    if use_wandb:
        wandb.init(
            name=wandb_name,
            project=WANDB_PROJECT,
            config={
                "Epochs": num_epochs,
                "Learning Rate": LEARNING_RATE,
                "Adam Betas": ADAM_BETAS,
                "Weight Decay": WEIGHT_DECAY,
                "Batch Size": BATCH_SIZE,
            },
        )

    # Load datasets
    training_set = NoisySpeechDataset(train=True, subsample=subsample)
    validation_set = NoisySpeechDataset(train=False, subsample=subsample)

    # Construct data loaders
    training_data_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3
    )
    validation_data_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3
    )

    # Initialize model
    net = WaveUNet().cuda() if use_cuda else WaveUNet().cpu()

    # Initialize optmizer
    optimizer = optim.AdamW(
        net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
    )

    # Keep track of loss history using moving average
    mean_squared_error = nn.MSELoss()
    training_mse = MovingAverage(decay=0.8)
    validation_mse = MovingAverage(decay=0.8)

    # Run training for some number of epochs.
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1} / {num_epochs}\n")
        torch.cuda.empty_cache()

        # Save checkpoint periodically.
        is_checkpoint_epoch = checkpoint_epochs and epoch % checkpoint_epochs == 0
        if CHECKPOINT_NAME and is_checkpoint_epoch:
            checkpoint.save(net, CHECKPOINT_NAME, name=wandb_name)

        # Run training loop
        net.train()
        for inputs, targets in tqdm(training_data_loader):
            inputs = inputs.cuda() if use_cuda else inputs.cpu()
            targets = targets.cuda() if use_cuda else targets.cpu()

            # 10% of the time, have no noise at all
            is_clean_input = random.random() <= 0.1
            if is_clean_input:
                inputs = targets

            # Add channel dimension to input
            batch_size = inputs.shape[0]
            audio_length = inputs.shape[1]
            inputs = inputs.view(batch_size, 1, -1)

            # Get a prediction from the model
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.squeeze(dim=1)
            assert outputs.shape == (batch_size, audio_length)

            # Run loss function on over the model's prediction
            assert targets.shape == (batch_size, audio_length)
            loss = mean_squared_error(outputs, targets)

            # Calculate model weight gradients from the loss and update model.
            loss.backward()
            optimizer.step()

            # Track training information
            mse = loss.data.item()
            training_mse.update(mse)

        # Check performance (loss) on validation set.
        net.eval()
        with torch.no_grad():
            for inputs, targets in tqdm(validation_data_loader):
                batch_size = inputs.shape[0]
                audio_length = inputs.shape[1]
                inputs = inputs.view(batch_size, 1, -1)
                inputs = inputs.cuda() if use_cuda else inputs.cpu()
                targets = targets.cuda() if use_cuda else targets.cpu()
                outputs = net(inputs)
                outputs = outputs.squeeze(dim=1)
                mse = mean_squared_error(outputs, targets).data.item()
                validation_mse.update(mse)

        log_training_info(
            {"Training Loss": training_mse.value, "Validation Loss": validation_mse.value},
            use_wandb=use_wandb,
        )

    # Save final model checkpoint
    if CHECKPOINT_NAME:
        checkpoint.save(net, CHECKPOINT_NAME, name=wandb_name, use_wandb=use_wandb)
