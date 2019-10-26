import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from ..datasets.speech_dataset import SpeechDataset
from ..models.wave_u_net import WaveUNet
from ..models.mel_discriminator import MelDiscriminatorNet
from ..models.scene_net import SceneNet
from ..utils.trackers import MovingAverage
from ..utils.loss import AudioFeatureLoss
from ..utils.checkpoint import save_checkpoint

DISC_NET_CHECKPOINT = None  # 'checkpoints/wave-u-net-gan-2-1572119346.ckpt'
LOSS_NET_CHECKPOINT = "checkpoints/scene-net-long-train.ckpt"
WANDB_NAME = None
USE_WANDB = True
USE_CUDA = True
NUM_EPOCHS = 8
CHECKPOINT_EPOCHS = 4
CHECKPOINT_NAME = "wave-u-net"
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
DISC_WEIGHT = 1e-2

if USE_WANDB:
    WANDB_NAME = WANDB_NAME or input("What do you want to call this run: ")
    WANDB_PROJECT = "wave-u-net"
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
training_data_loader = DataLoader(
    training_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=3, pin_memory=True
)
validation_data_loader = DataLoader(
    validation_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=3,
    pin_memory=True,
)

# Initialize model
net = WaveUNet().cuda() if USE_CUDA else WaveUNet().cpu()
# Initialize optmizer
optimizer = optim.AdamW(
    net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
)
if USE_WANDB:
    wandb.watch(net)

# Initialize feature loss function
loss_net = SceneNet().cuda()
state_dict = torch.load(LOSS_NET_CHECKPOINT)
loss_net.load_state_dict(state_dict)
loss_net.eval()
loss_net.set_feature_mode()
feature_loss_criterion = AudioFeatureLoss(loss_net)

# Initialize discriminator loss function, optimizer
disc_net = MelDiscriminatorNet().cuda() if USE_CUDA else MelDiscriminatorNet().cpu()
if DISC_NET_CHECKPOINT:
    state_dict = torch.load(LOSS_NET_CHECKPOINT)
    disc_net.load_state_dict(state_dict)

disc_net.train()
optimizer_disc = optim.AdamW(
    disc_net.parameters(),
    lr=LEARNING_RATE,
    betas=ADAM_BETAS,
    weight_decay=WEIGHT_DECAY,
)

# Keep track of loss history using moving average
disc_loss = MovingAverage(decay=0.8)
training_loss = MovingAverage(decay=0.8)
validation_loss = MovingAverage(decay=0.8)
mean_squared_error = nn.MSELoss()
training_mse = MovingAverage(decay=0.8)
validation_mse = MovingAverage(decay=0.8)

# ~2 min epoch
# Run training for some number of epochs.
for epoch in range(NUM_EPOCHS):
    torch.cuda.empty_cache()

    print(f"\nEpoch {epoch + 1} / {NUM_EPOCHS}\n")
    # Save checkpoint periodically.
    is_checkpoint_epoch = CHECKPOINT_EPOCHS and epoch % CHECKPOINT_EPOCHS == 0
    if CHECKPOINT_NAME and is_checkpoint_epoch:
        checkpoint_path = save_checkpoint(net, CHECKPOINT_NAME, name=WANDB_NAME)

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
        outputs = outputs.squeeze(dim=1)
        assert outputs.shape == (batch_size, audio_length)

        # Run loss function on over the model's prediction
        # TODO: Add discriminator to loss function
        targets = targets.cuda() if USE_CUDA else targets.cpu()
        assert targets.shape == (batch_size, audio_length)
        feature_loss = feature_loss_criterion(inputs, outputs, targets)

        # Add discriminator to loss function
        fake_audio = outputs.view(batch_size, 1, -1)
        disc_fake = disc_net(fake_audio)
        loss = feature_loss + DISC_WEIGHT * torch.mean((disc_fake - 1) ** 2)

        # Calculate model weight gradients from the loss
        loss.backward()

        # Update model weights via gradient descent.
        optimizer.step()

        # Track training information
        loss_amount = feature_loss.data.item()
        training_loss.update(loss_amount)
        mse = mean_squared_error(outputs, targets).data.item()
        training_mse.update(mse)

        # Train discriminator
        fake_audio = outputs.view(batch_size, 1, -1).detach()
        real_audio = inputs
        optimizer_disc.zero_grad()
        disc_fake = disc_net(fake_audio)
        disc_real = disc_net(real_audio)
        # 2 x (b, 1, 128)

        # Apply least squares loss
        loss = torch.mean((disc_real - 1) ** 2) + torch.mean(disc_fake ** 2)
        loss.backward()
        optimizer_disc.step()

        # Track disc training information
        loss_amount = loss.data.item()
        disc_loss.update(loss_amount)

    # Check performance (loss) on validation set.
    net.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(validation_data_loader):
            batch_size = inputs.shape[0]
            audio_length = inputs.shape[1]
            inputs = inputs.view(batch_size, 1, -1)
            inputs = inputs.cuda() if USE_CUDA else inputs.cpu()
            targets = targets.cuda() if USE_CUDA else targets.cpu()
            outputs = net(inputs)
            outputs = outputs.squeeze(dim=1)
            loss = feature_loss_criterion(inputs, outputs, targets)
            loss_amount = loss.data.item()
            validation_loss.update(loss_amount)
            mse = mean_squared_error(outputs, targets).data.item()
            validation_mse.update(mse)

    # Log training information for this epoch.
    training_info = {
        "Training Feature Loss": training_loss.value,
        "Validation Feature Loss": validation_loss.value,
        "Training Loss": training_mse.value,
        "Validation Loss": validation_mse.value,
        "Discriminator Loss": disc_loss.value,
    }
    print("")
    for k, v in training_info.items():
        s = "{k: <30}{v:0.4f}".format(k=k, v=v)
        print(s)
    if USE_WANDB:
        wandb.log(training_info)


# Save final model checkpoint
if CHECKPOINT_NAME:
    checkpoint_path = save_checkpoint(net, CHECKPOINT_NAME, name=WANDB_NAME)
    # Upload model to wandb
    if USE_WANDB:
        print(f"Uploading {checkpoint_path} to W&B")
        wandb.save(checkpoint_path)
