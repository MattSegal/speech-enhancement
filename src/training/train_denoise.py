"""
Training loop for speech denoiser
"""
import os
import time

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from ..datasets.speech_dataset import SpeechDataset
from ..models.denoise_net import SpeechDenoiseNet
from ..utils.moving_average import MovingAverage

# TODO - find good metric for regression
USE_CUDA = True
USE_WANDB = False
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 64
CHECKPOINT_DIR = "checkpoints"

if USE_WANDB:
    WANDB_NAME = input("What do you want to call this run: ")
    WANDB_PROJECT = "speech-denoise-net"
    wandb.init(
        name=WANDB_NAME or None,
        project=WANDB_PROJECT,
        config={
            "Epochs": NUM_EPOCHS,
            "Learning Rate": LEARNING_RATE,
            "Adam Betas": ADAM_BETAS,
            "Weight Decay": WEIGHT_DECAY,
            "Batch Size": BATCH_SIZE,
        },
    )

# Load datasets
training_set = SpeechDataset(train=True)
validation_set = SpeechDataset(train=False)

# Construct data loaders
training_data_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
validation_data_loader = DataLoader(
    validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=3
)

# Initialize model
net = SceneNet().cuda() if USE_CUDA else SceneNet().cpu()
if USE_WANDB:
    wandb.watch(net)

# TODO - rig up loss function properly
# criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
)

# Keep track of loss history using moving average
training_loss = MovingAverage(decay=0.8)
validation_loss = MovingAverage(decay=0.8)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1} / {NUM_EPOCHS}\n")

    # Run training loop
    net.train()
    for inputs, targets in tqdm(training_data_loader):
        # Add channel dimension to input
        batch_size = inputs.shape[0]
        audio_length = inputs.shape[1]
        inputs = inputs.view(batch_size, 1, -1)

        # Tell PyTorch to reset gradient tracking for new training run.
        optimizer.zero_grad()

        # Get a prediction from the model
        inputs = inputs.cuda() if USE_CUDA else inputs.cpu()
        outputs = net(inputs)
        assert outputs.shape == (batch_size, 1, audio_length)

        # Run loss function on over the model's prediction
        targets = targets.cuda() if USE_CUDA else targets.cpu()
        assert targets.shape == (batch_size, 1, audio_length)
        loss = criterion(outputs, targets)

        # Calculate model weight gradients from the loss
        loss.backward()

        # Update model weights via gradient descent.
        optimizer.step()

        # Track training information
        loss_amount = loss.data.item()
        training_loss.update(loss_amount)

    # Check performance (loss) on validation set.
    net.eval()
    for inputs, targets in tqdm(validation_data_loader):
        batch_size = inputs.shape[0]
        audio_length = inputs.shape[1]
        inputs = inputs.view(batch_size, 1, -1)
        inputs = inputs.cuda() if USE_CUDA else inputs.cpu()
        targets = targets.cuda() if USE_CUDA else targets.cpu()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss_amount = loss.data.item()
        validation_loss.update(loss_amount)

    # Log training information
    print(f"\n\tTraining loss:       {training_loss.value:0.4f}")
    print(f"\tValidation loss:     {validation_loss.value:0.4f}")
    print(f"\tTraining accuracy:   {training_accuracy.value:0.2f}")
    print(f"\tValidation accuracy: {validation_accuracy.value:0.2f}")

    if USE_WANDB:
        wandb.log(
            {
                "Training Loss": training_loss.value,
                "Validation Loss": validation_loss.value,
                "Training Accuracy": training_accuracy.value,
                "Validation Accuracy": validation_accuracy.value,
            }
        )

# Save model to disk
if USE_WANDB:
    checkpoint_filename = f"scene-net-{WANDB_NAME}-{int(time.time())}.ckpt"
else:
    checkpoint_filename = f"scene-net-{int(time.time())}.ckpt"

print(f"\nSaving checkpoint as {checkpoint_filename}\n")
checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
torch.save(net.state_dict(), checkpoint_path)

# Upload model to wandb
if USE_WANDB:
    wandb.save(checkpoint_path)