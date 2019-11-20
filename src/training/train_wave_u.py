import random

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from ..datasets import NoisyLibreSpeechDataset as Dataset
from ..models.wave_u_net import WaveUNet
from ..models.mel_discriminator import MelDiscriminatorNet
from ..utils.trackers import MovingAverage
from ..utils.loss import l1_loss, LeastSquaresLoss
from ..utils import checkpoint
from ..utils.log import log_training_info

# Checkpointing
CHECKPOINT_NAME = "wave-u-net"
WANDB_PROJECT = "wave-u-net"

# Training hyperparams
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-4
DISC_WEIGHT = 1e-1
DISC_LEARNING_RATE = 4 * LEARNING_RATE


def train(num_epochs, use_cuda, batch_size, wandb_name, subsample, checkpoint_epochs):
    use_wandb = bool(wandb_name)
    if use_wandb:
        wandb.init(
            name=wandb_name,
            project=WANDB_PROJECT,
            config={
                "Epochs": num_epochs,
                "Batch Size": batch_size,
                "Learning Rate": LEARNING_RATE,
                "Adam Betas": ADAM_BETAS,
                "Weight Decay": WEIGHT_DECAY,
            },
        )

    # Load datasets
    training_set = Dataset(train=True, subsample=subsample)
    validation_set = Dataset(train=False, subsample=subsample)

    # Construct data loaders
    training_data_loader = DataLoader(
        training_set, batch_size=batch_size, shuffle=True, num_workers=3
    )
    validation_data_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=True, num_workers=3
    )

    # Initialize model
    net = WaveUNet().cuda() if use_cuda else WaveUNet().cpu()

    # Initialize optmizer
    optimizer = optim.AdamW(
        net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
    )

    # Initialize discriminator loss function, optimizer
    disc_net = MelDiscriminatorNet().cuda() if use_cuda else MelDiscriminatorNet().cpu()

    disc_net.train()
    gan_loss = LeastSquaresLoss(disc_net)
    optimizer_disc = optim.AdamW(disc_net.parameters(), lr=DISC_LEARNING_RATE, betas=ADAM_BETAS)

    # Keep track of loss history using moving average
    disc_loss = MovingAverage(decay=0.8)
    gen_loss = MovingAverage(decay=0.8)
    training_loss = MovingAverage(decay=0.8)
    validation_loss = MovingAverage(decay=0.8)
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
            l1_loss_t = l1_loss(outputs, targets)

            # Add discriminator to loss function
            fake_audio = outputs.view(batch_size, 1, -1)
            gen_gan_loss = gan_loss.for_generator(inputs, fake_audio)

            loss = l1_loss_t + DISC_WEIGHT * gen_gan_loss

            # Calculate model weight gradients from the loss and update model.
            loss.backward()
            optimizer.step()

            # Track training information
            loss_amount = l1_loss_t.data.item()
            training_loss.update(loss_amount)
            mse = mean_squared_error(outputs, targets).data.item()
            training_mse.update(mse)
            loss_amount = gen_gan_loss.data.item()
            gen_loss.update(loss_amount)

            # Train discriminator
            optimizer_disc.zero_grad()
            fake_audio = outputs.view(batch_size, 1, -1).detach()
            disc_gan_loss = gan_loss.for_discriminator(inputs, fake_audio)
            disc_gan_loss.backward()
            optimizer_disc.step()

            # Track disc training information
            loss_amount = disc_gan_loss.data.item()
            disc_loss.update(loss_amount)

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
                loss = l1_loss(outputs, targets)
                loss_amount = loss.data.item()
                validation_loss.update(loss_amount)
                mse = mean_squared_error(outputs, targets).data.item()
                validation_mse.update(mse)

        log_training_info(
            {
                "Training L1 Loss": training_loss.value,
                "Validation L1 Loss": validation_loss.value,
                "Training Loss": training_mse.value,
                "Validation Loss": validation_mse.value,
                "Discriminator Loss": disc_loss.value,
                "Generator Loss": gen_loss.value,
            },
            use_wandb=use_wandb,
        )

    # Save final model checkpoint
    if CHECKPOINT_NAME:
        checkpoint.save(net, CHECKPOINT_NAME, name=wandb_name, use_wandb=use_wandb)
