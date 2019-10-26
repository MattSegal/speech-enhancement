import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb

from ..datasets.speech_dataset import SpeechDataset
from ..models.wave_u_net import WaveUNet
from ..models.scene_net import SceneNet
from ..utils.trackers import MovingAverage
from ..utils.loss import AudioFeatureLoss
from ..utils.checkpoint import save_checkpoint

LOSS_NET_CHECKPOINT = "checkpoints/scene-net-long-train.ckpt"
WANDB_NAME = None
USE_WANDB = True
USE_CUDA = True
NUM_EPOCHS = 12
CHECKPOINT_EPOCHS = 10
CHECKPOINT_NAME = "wave-u-net"
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0
BATCH_SIZE = 8

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
if USE_WANDB:
    wandb.watch(net)

# Initialize loss function, optimizer
loss_net = SceneNet().cuda()
state_dict = torch.load(LOSS_NET_CHECKPOINT)
loss_net.load_state_dict(state_dict)
loss_net.eval()
loss_net.set_feature_mode()
criterion = AudioFeatureLoss(loss_net)

optimizer = optim.AdamW(
    net.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
)

# Keep track of loss history using moving average
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
        targets = targets.cuda() if USE_CUDA else targets.cpu()
        assert targets.shape == (batch_size, audio_length)
        loss = criterion(inputs, outputs, targets)

        # Calculate model weight gradients from the loss
        loss.backward()

        # Update model weights via gradient descent.
        optimizer.step()

        # Track training information
        loss_amount = loss.data.item()
        training_loss.update(loss_amount)
        mse = mean_squared_error(outputs, targets).data.item()
        training_mse.update(mse)

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
            loss = criterion(inputs, outputs, targets)
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
