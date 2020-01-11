import pprint as pprint
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from src.utils import checkpoint
from src.utils.trackers import MovingAverage
from src.utils.log import log_training_info


class Trainer:
    def __init__(self, cuda):
        print("Initialising trainer...")
        # Training / runtime
        self.use_cuda = cuda
        self.scheduler = None

        # Checkpointing
        self.checkpoint_epochs = None
        self.checkpoint_name = None

        # Loss and metric tracking
        self.loss_fns = []
        self.metric_fns = []

        # Weight and Bias Logging
        self.wandb_name = None
        self.use_wandb = False

        # Shape assert
        self.input_shape = []
        self.target_shape = []
        self.output_shape = []

    def load_net(self, net_class, **kwargs):
        print(f"Loading net from {net_class}...")
        return net_class(**kwargs).cuda() if self.use_cuda else net_class(**kwargs).cpu()

    def setup_wandb(self, project_name, run_name, run_info):
        print("Using training config:")
        pprint.pprint(run_info)
        self.wandb_name = run_name
        self.use_wandb = bool(run_name)
        if self.use_wandb:
            print("Initializing W&B...")
            wandb.init(name=run_name, project=project_name, config=run_info)
        else:
            print("Skipping W&B init.")

    def setup_checkpoints(self, save_name, save_epochs):
        print("Setting up model checkpointing")
        self.checkpoint_name = save_name
        self.checkpoint_epochs = save_epochs

    def load_data_loaders(self, dataset, batch_size, subsample, **kwargs):
        print("Setting up datasets...")
        self.train_set = dataset(train=True, subsample=subsample, **kwargs)
        self.test_set = dataset(train=False, subsample=subsample, **kwargs)
        train_loader = self.load_data_loader(self.train_set, batch_size)
        test_loader = self.load_data_loader(self.test_set, batch_size)
        return train_loader, test_loader

    def load_data_loader(self, dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)

    def load_optimizer(self, net, learning_rate, adam_betas, weight_decay):
        print("Setting up optimizer...")
        return optim.AdamW(
            net.parameters(),
            lr=learning_rate,
            betas=adam_betas,
            weight_decay=weight_decay,
        )

    def use_cyclic_lr_scheduler(self, optimizer, step_size_up, base_lr, max_lr):
        """
        Use cyclic learning rate scheduler (CyclicLR).
        `step_size_up` is the number of training iterations in the increasing half of a cycle.
        See https://pytorch.org/docs/stable/optim.html
        """
        print("Setting up cycling learning rate scheduler...")
        self.scheduler = optim.lr_scheduler.CyclicLR(
            optimizer,
            step_size_up=step_size_up,
            base_lr=base_lr,
            max_lr=max_lr,
            cycle_momentum=False,
        )

    def use_one_cycle_lr_scheduler(self, optimizer, steps_per_epoch, epochs, max_lr):
        """
        Use "one cycle" learning rate scheduler (OneCycleLR).
        `steps_per_epoch` is the number of training iterations in the increasing half of a cycle.
        See https://pytorch.org/docs/stable/optim.html
        """
        print("Setting up 1-cycle learning rate scheduler...")
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, epochs=epochs, steps_per_epoch=steps_per_epoch, max_lr=max_lr,
        )

    def register_loss_fn(self, fn, weight=1):
        self.loss_fns.append([fn, weight])

    def register_metric_fn(self, fn, name):
        test_tracker = MovingAverage(decay=0.8)
        train_tracker = MovingAverage(decay=0.8)
        self.metric_fns.append([fn, name.capitalize(), train_tracker, test_tracker])

    def train(self, net, num_epochs, optimizer, train_loader, test_loader):
        print("Starting training...")
        # Run training for some number of epochs.
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1} / {num_epochs}\n")
            torch.cuda.empty_cache()

            # Save checkpoint periodically.
            is_checkpoint_epoch = (
                self.checkpoint_epochs and epoch % self.checkpoint_epochs == 0
            )
            if self.checkpoint_name and is_checkpoint_epoch:
                checkpoint.save(net, self.checkpoint_name, name=self.wandb_name)

            # Run training loop
            net.train()
            for inputs, targets in tqdm(train_loader):
                batch_size = inputs.shape[0]
                inputs = inputs.cuda() if self.use_cuda else inputs.cpu()
                targets = targets.cuda() if self.use_cuda else targets.cpu()

                # Sanity check training data shape sizes
                if self.input_shape:
                    expected_shape = tuple([batch_size] + self.input_shape)
                    assert (
                        inputs.shape == expected_shape
                    ), f"Bad shape: expected {expected_shape} got {inputs.shape}"
                if self.target_shape:
                    expected_shape = tuple([batch_size] + self.target_shape)
                    assert (
                        targets.shape == expected_shape
                    ), f"Bad shape: expected {expected_shape} got {target.shape}"

                # Get a prediction from the model
                optimizer.zero_grad()
                outputs = net(inputs)
                if self.output_shape:
                    expected_shape = tuple([batch_size] + self.output_shape)
                    assert (
                        outputs.shape == expected_shape
                    ), f"Bad shape: expected {expected_shape} got {outputs.shape}"

                # Run loss function on over the model's prediction
                loss = torch.tensor([0.0], requires_grad=True)
                loss = loss.cuda() if self.use_cuda else loss
                for loss_fn, weight in self.loss_fns:
                    loss = loss + weight * loss_fn(inputs, outputs, targets)

                # Calculate model weight gradients from the loss and update model.
                loss.backward()
                optimizer.step()
                if self.scheduler:
                    # Update the learning rate, according to the scheduler.
                    try:
                        self.scheduler.step()
                    except ValueError:
                        pass

                # Track metric information
                with torch.no_grad():
                    for metric_fn, _, train_tracker, _ in self.metric_fns:
                        metric_val = metric_fn(inputs, outputs, targets)
                        train_tracker.update(metric_val)

            # Check performance (loss) on validation set.
            net.eval()
            with torch.no_grad():
                for inputs, targets in tqdm(test_loader):
                    inputs = inputs.cuda() if self.use_cuda else inputs.cpu()
                    targets = targets.cuda() if self.use_cuda else targets.cpu()
                    outputs = net(inputs)
                    # Track metric information
                    for metric_fn, _, _, test_tracker in self.metric_fns:
                        metric_val = metric_fn(inputs, outputs, targets)
                        test_tracker.update(metric_val)

            # Log epoch metrics
            training_info = {}
            for _, name, train_tracker, test_tracker in self.metric_fns:
                training_info[f"Training {name}"] = train_tracker.value
                training_info[f"Validation {name}"] = test_tracker.value

            if self.scheduler:
                try:
                    training_info[f"Learning rate"] = self.scheduler.get_lr()[0]
                except ValueError:
                    pass  # Whatevs

            log_training_info(training_info, use_wandb=self.use_wandb)

        # Save final model checkpoint
        if self.checkpoint_name:
            checkpoint.save(
                net, self.checkpoint_name, name=self.wandb_name, use_wandb=self.use_wandb
            )
