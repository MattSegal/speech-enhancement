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
BATCH_SIZE = 128
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
    use_chime = epoch % 2 == 0
    if use_chime:
        print("Using CHiME dataset")
        net.set_chime_dataset()
        training_set = chime_training_set
        validation_set = chime_validation_set
        data_loader = chime_training_data_loader
        validation_data_loader = chime_validation_data_loader
        training_accuracy = HammingLossTracker(len(training_set), 8)
        validation_accuracy = HammingLossTracker(len(validation_set), 8)
        training_loss = chime_training_loss
        validation_loss = chime_validation_loss
        criterion = chime_criterion
    else:
        print("Using TUT dataset")
        net.set_tut_dataset()
        training_set = tut_training_set
        validation_set = tut_validation_set
        data_loader = tut_training_data_loader
        validation_data_loader = tut_validation_data_loader
        training_accuracy = AccuracyTracker(len(training_set))
        validation_accuracy = AccuracyTracker(len(validation_set))
        training_loss = tut_training_loss
        validation_loss = tut_validation_loss
        criterion = tut_criterion

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
    net.eval()
    for inputs, labels in tqdm(validation_data_loader):
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
