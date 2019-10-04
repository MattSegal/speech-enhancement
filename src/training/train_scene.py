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
from ..models.scene_net import SceneNet
from ..utils.moving_average import MovingAverage
from ..utils.accuracy_tracker import AccuracyTracker

USE_WANDB = True
NUM_EPOCHS = 12
LEARNING_RATE = 0.0001
ADAM_BETAS = (0.9, 0.999)
BATCH_SIZE = 128
CHECKPOINT_DIR = "checkpoints"

if USE_WANDB:
    WANDB_NAME = input("What do you want to call this run: ")
    WANDB_PROJECT = "speech-denoising-deep-feature-loss-scene-net"
    wandb.init(
        name=WANDB_NAME or None,
        project=WANDB_PROJECT,
        config={
            "Epochs": NUM_EPOCHS,
            "Learning Rate": LEARNING_RATE,
            "Adam Betas": ADAM_BETAS,
            "Batch Size": BATCH_SIZE,
        },
    )

# Load dataset
training_set = SceneDataset(train=True)
validation_set = SceneDataset(train=False)
num_labels = len(training_set.labels)

# Initialize model
net = SceneNet(num_labels=num_labels).cuda()
if USE_WANDB:
    wandb.watch(net)

# Setup loss function, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

# Construct data loaders
data_loader = DataLoader(
    training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3
)
validation_data_loader = DataLoader(
    validation_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=3
)

# Keep track of loss history using moving average
training_loss = MovingAverage(decay=0.8)
validation_loss = MovingAverage(decay=0.8)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1} / {NUM_EPOCHS}\n")
    training_accuracy = AccuracyTracker(len(training_set))

    # Run training loop
    net.train()
    for inputs, labels in tqdm(data_loader):
        # Add channel dimension to input
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, 1, -1)

        # Tell PyTorch to reset gradient tracking for new training run.
        optimizer.zero_grad()

        # Get a prediction from the model
        outputs = net(inputs.cuda())
        assert outputs.shape == (batch_size, num_labels)
        training_accuracy.update(outputs, labels.cuda())

        # Run loss function on over the model's prediction
        assert labels.shape == (batch_size,)
        loss = criterion(outputs, labels.cuda())

        # Calculate model weight gradients from the loss
        loss.backward()

        # Update model weights via gradient descent.
        optimizer.step()

        # Track training information
        loss_amount = loss.data.item()
        training_loss.update(loss_amount)

    # Check performance (loss, accurancy) on validation set.
    net.eval()
    validation_accuracy = AccuracyTracker(len(validation_set))
    for inputs, labels in tqdm(validation_data_loader):
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, 1, -1)
        # torch.Size([256, 1, 32767])
        outputs = net(inputs.cuda())
        # torch.Size([256, 15])
        validation_accuracy.update(outputs, labels.cuda())
        loss = criterion(outputs, labels.cuda())
        loss_amount = loss.data.item()
        validation_loss.update(loss_amount)

    # Log training information
    print(f"\n\tTraining loss:       {training_loss.value:0.4f}")
    print(f"\tValidation loss:     {validation_loss.value:0.4f}")
    print(f"\tTraining accuracy:   {training_accuracy.accuracy:0.2f}")
    print(f"\tValidation accuracy: {validation_accuracy.accuracy:0.2f}")
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
checkpoint_filename = f"scene-net-{int(time.time())}.ckpt"
checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_filename)
torch.save(net.state_dict(), checkpoint_path)
# Upload model to wandb
if USE_WANDB:
    wandb.save(checkpoint_path)
