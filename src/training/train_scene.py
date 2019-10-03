import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from ..datasets.scene_dataset import SceneDataset
from ..models.scene_net import SceneNet
from ..utils.loss_tracker import LossTracker
from ..utils.accuracy_tracker import AccuracyTracker

NUM_EPOCHS = 5  # 2500 in paper
LEARNING_RATE = 0.0001
ADAM_BETAS = (0.9, 0.999)
BATCH_SIZE = 64

WANDB_PROJECT = "speech-denoising-deep-feature-loss-scene-net"
# wandb.init(
#     project=WANDB_PROJECT,
#     config={
#         "Epochs": NUM_EPOCHS,
#         "Learning Rate": LEARNING_RATE,
#         "Adam Betas": ADAM_BETAS,
#         "Batch Size": BATCH_SIZE,
#     },
# )

# Load dataset
training_set = SceneDataset(train=True)
validation_set = SceneDataset(train=False)
num_labels = len(training_set.labels)

# Initialize model
net = SceneNet(num_labels=num_labels).cuda()
# wandb.watch(net)
net.train()

# Setup loss function, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

# Construct data loaders
data_loader = DataLoader(
    training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3
)
validation_data_loader = DataLoader(
    validation_set, batch_size=4, shuffle=False, num_workers=3
)

# Keep track of loss history using moving average
training_loss = LossTracker(decay=0.8)
validation_loss = LossTracker(decay=0.8)

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1} / {NUM_EPOCHS}\n")
    # Run training loop
    for inputs, labels in tqdm(data_loader):
        # Add channel dimension to input
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, 1, -1)

        # Tell PyTorch to reset gradient tracking for new training run.
        optimizer.zero_grad()

        # Get a prediction from the model
        outputs = net(inputs.cuda())
        assert outputs.shape == (batch_size, num_labels)

        # Run loss function on over the model's prediction
        loss = criterion(outputs.cuda(), labels.cuda())

        # Calculate model weight gradients from the loss
        loss.backward()

        # Update model weights via gradient descent.
        optimizer.step()

        # Track training information
        loss_amount = loss.data.item()
        training_loss.update(loss_amount)

    # Check performance (loss, accurancy) on validation set.
    accuracy = AccuracyTracker(len(validation_set))
    for inputs, labels in tqdm(validation_data_loader):
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, 1, -1)
        outputs = net(inputs.cuda())
        accuracy.update(outputs, labels.cuda())
        loss = criterion(outputs.cuda(), labels.cuda())
        loss_amount = loss.data.item()
        validation_loss.update(loss_amount)

    # Log training information
    print(f"\n\tTraining loss:       {training_loss.loss:0.4f}")
    print(f"\tValidation loss:     {validation_loss.loss:0.4f}")
    print(f"\tValidation accuracy: {accuracy.accuracy:0.2f}")
    # wandb.log(
    #     {
    #         "Training Loss": training_loss.loss,
    #         "Validation Loss": validation_loss.loss,
    #         "Validation Accuracy": accuracy.accuracy,
    #     }
    # )

# TODO - save model checkpoints to wandb
