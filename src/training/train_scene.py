"""
Training loop for acoustic scene classifier

Training history available on Weights and Biases
https://app.wandb.ai/mattdsegal/speech-denoising-deep-feature-loss-scene-net
"""
import os
import time

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from ..datasets.scene_dataset import SceneDataset
from ..datasets.chime_dataset import ChimeDataset
from ..models.scene_net import SceneNet
from ..utils.moving_average import MovingAverage
from ..utils.accuracy_tracker import AccuracyTracker, HammingLossTracker

USE_CUDA = True
USE_WANDB = False
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 64
CHECKPOINT_DIR = "checkpoints"

if USE_WANDB:
    WANDB_NAME = input("What do you want to call this run: ")
    # WANDB_PROJECT = "speech-denoising-deep-feature-loss-scene-net"
    WANDB_PROJECT = "chime-scene-net"
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

# Load dataset
training_set = ChimeDataset(train=True)
validation_set = ChimeDataset(train=False)

# Initialize model
net = SceneNet().cuda() if USE_CUDA else SceneNet().cpu()
net.set_chime_dataset()
if USE_WANDB:
    wandb.watch(net)

# Setup loss functions, optimizer
tut_criterion = nn.CrossEntropyLoss()
chime_criterion = nn.BCELoss()

optimizer = optim.AdamW(
    net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
)

# Construct data loaders
data_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
validation_data_loader = DataLoader(
    validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=3
)

# Keep track of loss history using moving average
training_loss = MovingAverage(decay=0.8)
validation_loss = MovingAverage(decay=0.8)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1} / {NUM_EPOCHS}\n")
    training_accuracy = HammingLossTracker(len(training_set), 8)

    # Run training loop
    net.train()
    for inputs, labels in tqdm(data_loader):
        # Add channel dimension to input
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, 1, -1)

        # Tell PyTorch to reset gradient tracking for new training run.
        optimizer.zero_grad()

        # Get a prediction from the model
        inputs = inputs.cuda() if USE_CUDA else inputs.cpu()
        outputs = net(inputs)
        assert outputs.shape == (batch_size, net.num_labels)

        # Run loss function on over the model's prediction
        labels = labels.cuda() if USE_CUDA else labels.cpu()
        if net.dataset == net.TUT:
            # Expect an array of indexes
            assert labels.shape == (batch_size,)
            loss = tut_criterion(outputs, labels)
        else:
            # Expect an array of class label vectors
            assert labels.shape == (batch_size, net.num_labels)
            loss = chime_criterion(outputs, labels)

        # Calculate model weight gradients from the loss
        loss.backward()

        # Update model weights via gradient descent.
        optimizer.step()

        # Track training information
        loss_amount = loss.data.item()
        training_loss.update(loss_amount)
        training_accuracy.update(outputs, labels)

    # Check performance (loss, accurancy) on validation set.
    net.eval()
    validation_accuracy = HammingLossTracker(len(validation_set), 8)
    for inputs, labels in tqdm(validation_data_loader):
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, 1, -1)
        inputs = inputs.cuda() if USE_CUDA else inputs.cpu()
        labels = labels.cuda() if USE_CUDA else labels.cpu()
        outputs = net(inputs)
        if net.dataset == net.TUT:
            loss = tut_criterion(outputs, labels)
        else:
            loss = chime_criterion(outputs, labels)

        loss_amount = loss.data.item()
        validation_loss.update(loss_amount)
        validation_accuracy.update(outputs, labels)

    # Log training information
    print(f"\n\tTraining loss:       {training_loss.value:0.4f}")
    print(f"\tValidation loss:     {validation_loss.value:0.4f}")
    print(f"\tTraining accuracy:   {training_accuracy.hamming_loss:0.2f}")
    print(f"\tValidation accuracy: {validation_accuracy.hamming_loss:0.2f}")
    if USE_WANDB:
        wandb.log(
            {
                "Training Loss": training_loss.value,
                "Validation Loss": validation_loss.value,
                "Training Accuracy": training_accuracy.accuracy,
                "Validation Accuracy": validation_accuracy.accuracy,
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
