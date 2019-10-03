import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..datasets.scene_dataset import SceneDataset
from ..models.scene_net import SceneNet

NUM_EPOCHS = 5  # 2500 in paper
LEARNING_RATE = 0.001
ADAM_BETAS = (0.9, 0.999)

# Load dataset
# TODO - load validation data for real training
data_set = SceneDataset(train=True)
num_labels = len(data_set.labels)

# Initialize model
net = SceneNet(num_labels=num_labels).cuda()

# Setup loss function, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)

# Construct data loader
BATCH_SIZE = 128
data_loader = DataLoader(
    data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3
)

# Keep track of loss history using moving average
loss_avg = 0
loss_beta = 0.8

# Run training loop
for epoch in range(NUM_EPOCHS):
    # Naievly loop through dataset
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
        loss_avg = loss_beta * loss_avg + (1 - loss_beta) * loss_amount

    # TODO - track accuracy metric

    # Log training information
    print(f"\nEpoch {epoch + 1} / {NUM_EPOCHS}")
    print(f"Loss: {loss_avg:0.4f}")

    # TODO - log accuracy on validation set!
