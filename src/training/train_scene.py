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
from ..utils.data_load import CombinedDataLoader

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

# Load datasets
chime_training_set = ChimeDataset(train=True)
chime_validation_set = ChimeDataset(train=False)
tut_training_set = SceneDataset(train=True)
tut_validation_set = SceneDataset(train=False)

# Construct data loaders
chime_training_data_loader = DataLoader(
    chime_training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3
)
chime_validation_data_loader = DataLoader(
    chime_validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=3
)
tut_training_data_loader = DataLoader(
    tut_training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3
)
tut_validation_data_loader = DataLoader(
    tut_validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=3
)

# Initialize model
net = SceneNet().cuda() if USE_CUDA else SceneNet().cpu()
if USE_WANDB:
    wandb.watch(net)

# Setup loss functions, optimizer
tut_criterion = nn.CrossEntropyLoss()
chime_criterion = nn.BCELoss()
optimizer = optim.AdamW(
    net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
)

# Keep track of loss history using moving average
chime_training_loss = MovingAverage(decay=0.8)
chime_validation_loss = MovingAverage(decay=0.8)
tut_training_loss = MovingAverage(decay=0.8)
tut_validation_loss = MovingAverage(decay=0.8)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1} / {NUM_EPOCHS}\n")
    data_loader = CombinedDataLoader(chime_training_data_loader, tut_training_data_loader)

    num_samples = min([len(chime_training_set), len(tut_training_set)])
    chime_training_accuracy = HammingLossTracker(num_samples, 8)
    chime_validation_accuracy = HammingLossTracker(num_samples, 8)
    tut_training_accuracy = AccuracyTracker(num_samples)
    tut_validation_accuracy = AccuracyTracker(num_samples)

    # Run training loop
    net.train()
    for inputs, labels in tqdm(data_loader):
        use_chime = data_loader.is_loader_a
        if use_chime:
            net.set_chime_dataset()
            training_accuracy = chime_training_accuracy
            training_loss = chime_training_loss
            criterion = chime_criterion
        else:
            net.set_tut_dataset()
            training_accuracy = tut_training_accuracy
            training_loss = tut_training_loss
            criterion = tut_criterion

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
        expect_labels_shape = (batch_size, net.num_labels) if use_chime else (batch_size,)
        assert labels.shape == expect_labels_shape
        loss = criterion(outputs, labels)

        # Calculate model weight gradients from the loss
        loss.backward()

        # Update model weights via gradient descent.
        optimizer.step()

        # Track training information
        loss_amount = loss.data.item()
        training_loss.update(loss_amount)
        training_accuracy.update(outputs, labels)

    # Check performance (loss, accurancy) on validation set.
    data_loader = CombinedDataLoader(chime_validation_data_loader, tut_validation_data_loader)
    net.eval()
    for inputs, labels in tqdm(data_loader):
        use_chime = data_loader.is_loader_a
        if use_chime:
            net.set_chime_dataset()
            validation_accuracy = chime_validation_accuracy
            validation_loss = chime_validation_loss
            criterion = chime_criterion
        else:
            net.set_tut_dataset()
            validation_accuracy = tut_validation_accuracy
            validation_loss = tut_validation_loss
            criterion = tut_criterion

        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, 1, -1)
        inputs = inputs.cuda() if USE_CUDA else inputs.cpu()
        labels = labels.cuda() if USE_CUDA else labels.cpu()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss_amount = loss.data.item()
        validation_loss.update(loss_amount)
        validation_accuracy.update(outputs, labels)

    # Log training information
    print("\nCHiME training data")
    print(f"\n\tTraining loss:       {chime_training_loss.value:0.4f}")
    print(f"\tValidation loss:     {chime_validation_loss.value:0.4f}")
    print(f"\tTraining accuracy:   {chime_training_accuracy.value:0.2f}")
    print(f"\tValidation accuracy: {chime_validation_accuracy.value:0.2f}")

    print("\nTUT training data")
    print(f"\n\tTraining loss:       {tut_training_loss.value:0.4f}")
    print(f"\tValidation loss:     {tut_validation_loss.value:0.4f}")
    print(f"\tTraining accuracy:   {tut_training_accuracy.value:0.2f}")
    print(f"\tValidation accuracy: {tut_validation_accuracy.value:0.2f}")

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
