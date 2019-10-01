import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from .progress_bar import ProgressBar


class Trainer:
    def __init__(self, net, data_set, num_epochs, **kwargs):
        """
        net         - a trainable model (torch.nn.Module)
        data_set    - training data (torch.utils.data.DataLoader) 
        num_epochs  - number of epochs to train for
        """
        self.net = net
        self.data_set = data_set
        self.num_epochs = num_epochs

        # Get hyper-parameters, set defaults
        learning_rate = kwargs.get("learning_rate", 0.001)
        self.batch_size = kwargs.get("batch_size", 128)

        # Use cross-entropy loss since we're doing multi-class classification
        self.criterion = kwargs.get("criterion", nn.CrossEntropyLoss())

        # Use Adam optimizer, because it's faster than classic gradient descent.
        adam_betas = kwargs.get("adam_betas", (0.9, 0.999))
        self.optimizer = optim.Adam(
            net.parameters(), lr=learning_rate, betas=adam_betas
        )

        # Construct data loader
        shuffle = kwargs.get("shuffle", True)
        sampler = kwargs.get("sampler", None)
        self.data_loader = DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=3,
        )

    def train(self):
        """Train model"""
        # Keep track of loss history
        loss_history = []
        loss_avg = 0
        loss_beta = 0.8

        # Run training loop
        for epoch in range(self.num_epochs):
            with ProgressBar(epoch, self.num_epochs, self.data_loader) as progress:
                for inputs, labels in progress:
                    # Tell PyTorch to reset gradient tracking.
                    self.optimizer.zero_grad()

                    # Get prediction from model
                    outputs = self.net(inputs.cuda())

                    # Run loss function on over the model's prediction
                    loss = self.criterion(outputs.cuda(), labels.cuda())

                    # Calculate model weight gradients from the loss
                    loss.backward()

                    # Update model weights via gradient descent.
                    self.optimizer.step()

                    # Log training information
                    loss_amount = loss.data.item()
                    loss_avg = loss_beta * loss_avg + (1 - loss_beta) * loss_amount
                    progress.update(loss_avg)
                    loss_history.append(loss_amount)

        # Plot training performance
        fig, ax = plt.subplots()
        ax.plot(loss_history)
        ax.set_title("Training performance")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")

