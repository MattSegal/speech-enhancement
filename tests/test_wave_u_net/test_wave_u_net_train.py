from unittest import mock

import numpy as np
import torch

from src.tasks.waveunet.training.train_mse import train as train_mse

from tests.utils import DummyNet

INPUT_SHAPE = (1, 2 ** 15)
OUTPUT_SHAPE = (1, 2 ** 15)
USE_CUDA = torch.cuda.is_available()


@mock.patch("src.utils.trainer.checkpoint", autospec=True)
def test_train_mse_no_model(mock_checkpoint):
    """
    Check that training loop runs without crashing, when there is no model
    """
    dummy_net = DummyNet(INPUT_SHAPE, OUTPUT_SHAPE, USE_CUDA)
    with mock.patch("src.tasks.waveunet.training.train_mse.WaveUNet") as net_cls:
        net_cls.return_value = dummy_net
        train_mse(
            num_epochs=2,
            use_cuda=USE_CUDA,
            batch_size=1,
            wandb_name=None,
            subsample=4,
            checkpoint_epochs=None,
        )


@mock.patch("src.utils.trainer.checkpoint", autospec=True)
def test_train_mse_with_model(mock_checkpoint):
    """
    Check that training loop runs without crashing, when there is a model
    """
    train_mse(
        num_epochs=2,
        use_cuda=USE_CUDA,
        batch_size=1,
        wandb_name=None,
        subsample=4,
        checkpoint_epochs=None,
    )
