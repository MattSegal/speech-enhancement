"""
Train UNet on a spectrogram of the noisy VCTK dataset using feature loss
"""
import torch.nn as nn

from src.datasets import NoisySpectralSpeechDataset as Dataset
from src.utils.trainer import Trainer
from src.utils.loss import AudioFeatureLoss
from src.utils.checkpoint import load as load_checkpoint

from .model import SpectralUNet

# Hyperparams
MIN_LR = 2e-4
MAX_LR = 1e-3
ADAM_BETAS = (0.9, 0.99)
WEIGHT_DECAY = 1e-6

LOSS_NET_CHECKPOINT = "spectral-scene-net-spec-scenes-6-1577253227.full.ckpt"

mse = nn.MSELoss()


def train(runtime, training, logging):

    # Load feature loss net
    loss_net = load_checkpoint(LOSS_NET_CHECKPOINT, use_cuda=runtime["cuda"])
    loss_net.set_feature_mode(num_layers=6)
    loss_net.eval()
    feature_loss = AudioFeatureLoss(loss_net, use_cuda=runtime["cuda"])

    def get_feature_loss(inputs, outputs, targets):
        return feature_loss(inputs, outputs, targets)

    def get_feature_loss_metric(inputs, outputs, targets):
        loss_t = feature_loss(inputs, outputs, targets)
        return loss_t.data.item()

    batch_size = training["batch_size"]
    epochs = training["epochs"]
    subsample = training["subsample"]
    trainer = Trainer(**runtime)
    trainer.setup_checkpoints(**logging["checkpoint"])
    trainer.setup_wandb(
        **logging["wandb"],
        run_info={
            "Batch Size": batch_size,
            "Epochs": epochs,
            "Adam Betas": ADAM_BETAS,
            "Learning Rate": [MIN_LR, MAX_LR],
            "Weight Decay": WEIGHT_DECAY,
            "Fine Tuning": False,
        }
    )

    train_loader, test_loader = trainer.load_data_loaders(Dataset, batch_size, subsample)

    trainer.register_loss_fn(get_feature_loss)
    trainer.register_metric_fn(get_mse_metric, "Loss")
    trainer.register_metric_fn(get_feature_loss_metric, "Feature Loss")

    trainer.input_shape = [1, 80, 256]
    trainer.target_shape = [1, 80, 256]
    trainer.output_shape = [1, 80, 256]
    net = trainer.load_net(SpectralUNet)

    optimizer = trainer.load_optimizer(
        net, learning_rate=MIN_LR, adam_betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY
    )
    steps_per_epoch = len(trainer.train_set) // batch_size
    trainer.use_one_cycle_lr_scheduler(optimizer, steps_per_epoch, epochs, MAX_LR)

    trainer.train(net, epochs, optimizer, train_loader, test_loader)


def get_mse_loss(inputs, outputs, targets):
    return mse(outputs, targets)


def get_mse_metric(inputs, outputs, targets):
    mse_t = mse(outputs, targets)
    return mse_t.data.item()
