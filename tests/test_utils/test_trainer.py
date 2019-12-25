from unittest import mock

import torch
import numpy as np
from torch import nn

from src.tasks.spectral_u_net.train import train
from src.utils.trainer import Trainer

from tests.utils import DummyNet, DummyDataset

INPUT_SHAPE = (1, 80, 256)
OUTPUT_SHAPE = (1, 80, 256)
USE_CUDA = torch.cuda.is_available()

mse = nn.MSELoss()


@mock.patch("src.utils.trainer.checkpoint", autospec=True)
def test_train(mock_checkpoint):
    """
    Check that training loop runs without crashing, when there is no model
    """
    trainer = Trainer(use_cuda=USE_CUDA, wandb_name="my-model")
    trainer.setup_checkpoints("my-checkpoint", checkpoint_epochs=None)
    train_loader, test_loader = trainer.load_data_loaders(
        DummyDataset,
        batch_size=16,
        subsample=None,
        build_output=_build_output,
        length=128,
    )
    trainer.register_loss_fn(_get_mse_loss)
    trainer.register_metric_fn(_get_mse_metric, "Loss")
    trainer.input_shape = [1, 80, 256]
    trainer.target_shape = [1, 80, 256]
    trainer.output_shape = [1, 80, 256]
    net = trainer.load_net(
        DummyNet,
        input_shape=(16,) + INPUT_SHAPE,
        output_shape=(16,) + OUTPUT_SHAPE,
        use_cuda=USE_CUDA,
    )
    optimizer = trainer.load_optimizer(
        net, learning_rate=1e-4, adam_betas=[0.9, 0.99], weight_decay=1e-6
    )
    epochs = 3
    mock_checkpoint.save.assert_not_called()
    trainer.train(net, epochs, optimizer, train_loader, test_loader)
    mock_checkpoint.save.assert_called_once_with(
        net, "my-checkpoint", name="my-model", use_wandb=False
    )


@mock.patch("src.utils.trainer.checkpoint", autospec=True)
def test_train_with_lr_scheduler(mock_checkpoint):
    """
    Check that training loop runs without crashing, when there is no model
    and when there is a learning rate scheulder used
    """
    trainer = Trainer(use_cuda=USE_CUDA, wandb_name="my-model")
    trainer.setup_checkpoints("my-checkpoint", checkpoint_epochs=None)
    train_loader, test_loader = trainer.load_data_loaders(
        DummyDataset,
        batch_size=16,
        subsample=None,
        build_output=_build_output,
        length=128,
    )
    trainer.register_loss_fn(_get_mse_loss)
    trainer.register_metric_fn(_get_mse_metric, "Loss")
    trainer.input_shape = [1, 80, 256]
    trainer.target_shape = [1, 80, 256]
    trainer.output_shape = [1, 80, 256]
    net = trainer.load_net(
        DummyNet,
        input_shape=(16,) + INPUT_SHAPE,
        output_shape=(16,) + OUTPUT_SHAPE,
        use_cuda=USE_CUDA,
    )
    optimizer = trainer.load_optimizer(
        net, learning_rate=1e-4, adam_betas=[0.9, 0.99], weight_decay=1e-6
    )
    epochs = 5

    # One cycle learning rate
    steps_per_epoch = len(trainer.train_set) // 16
    trainer.use_one_cycle_lr_scheduler(optimizer, steps_per_epoch, epochs, 1e-3)

    mock_checkpoint.save.assert_not_called()
    trainer.train(net, epochs, optimizer, train_loader, test_loader)
    mock_checkpoint.save.assert_called_once_with(
        net, "my-checkpoint", name="my-model", use_wandb=False
    )


def _get_mse_loss(inputs, outputs, targets):
    return mse(outputs, targets)


def _get_mse_metric(inputs, outputs, targets):
    mse_t = mse(outputs, targets)
    return mse_t.data.item()


def _build_output():
    input_t = torch.Tensor(np.random.random(INPUT_SHAPE))
    target_t = torch.Tensor(np.random.random(OUTPUT_SHAPE))
    return input_t, target_t
