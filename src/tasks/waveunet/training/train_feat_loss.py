"""
Train WaveUNet on the noisy VCTK dataset using feature loss + MSE

Batch size of 8 uses approx ??GB of GPU memory.

FIXME: ADD THIS BACK - use new scenes model
"""
import torch.nn as nn


from src.datasets import NoisySpeechDataset as Dataset
from src.utils.loss import AudioFeatureLoss
from src.utils.trainer import Trainer
from src.utils.checkpoint import load as load_checkpoint
from ..models.wave_u_net import WaveUNet

# Checkpointing
WANDB_PROJECT = "wave-u-net"
CHECKPOINT_NAME = "wave-u-net"

# Loss net
LOSS_NET_CHECKPOINT = "scene-net-long-train.ckpt"

# Training hyperparams
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-4


mse = nn.MSELoss()


def train(num_epochs, use_cuda, batch_size, wandb_name, subsample, checkpoint_epochs):

    # Load loss net
    loss_net = load_checkpoint(LOSS_NET_CHECKPOINT, use_cuda=use_cuda)
    loss_net.set_feature_mode()
    loss_net.eval()

    feature_loss = AudioFeatureLoss(loss_net, use_cuda=use_cuda)

    def get_feature_loss(inputs, outputs, targets):
        return feature_loss(outputs, targets)

    def get_feature_loss_metric(inputs, outputs, targets):
        loss_t = feature_loss(outputs, targets)
        return loss_t.data.item()

    trainer = Trainer(num_epochs, wandb_name)
    trainer.setup_checkpoints(CHECKPOINT_NAME, checkpoint_epochs)
    trainer.setup_wandb(WANDB_PROJECT, wandb_name)
    noise_data = NoisyScenesDataset(subsample=subsample)
    train_loader, test_loader = trainer.load_data_loaders(
        Dataset, batch_size, subsample, noise_data=noise_data
    )
    trainer.register_loss_fn(get_feature_loss)
    trainer.register_metric_fn(get_mse_metric, "Loss")
    trainer.register_metric_fn(get_mse_metric, "Feature Loss")
    trainer.input_shape = [2 ** 15]
    trainer.target_shape = [2 ** 15]
    trainer.output_shape = [2 ** 15]
    net = trainer.load_net(WaveUNet)
    optimizer = trainer.load_optimizer(
        net,
        learning_rate=LEARNING_RATE,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    trainer.train(net, num_epochs, optimizer, train_loader, test_loader)
    # Do a fine tuning run with 1/10th learning rate for 1/3rd epochs.
    optimizer = trainer.load_optimizer(
        net,
        learning_rate=LEARNING_RATE / 10,
        adam_betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY / 10,
    )
    num_epochs = num_epochs // 3
    trainer.train(net, num_epochs, optimizer, train_loader, test_loader)


def get_mse_metric(inputs, outputs, targets):
    mse_t = mse(outputs, targets)
    return mse_t.data.item()
