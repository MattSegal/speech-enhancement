from unittest import mock

import numpy as np
import torch

from src.tasks.spectral_u_net.train import train

from tests.utils import DummyNet

INPUT_SHAPE = (1, 2, 256, 128)
OUTPUT_SHAPE = (1, 2, 256, 128)


@mock.patch("src.utils.trainer.checkpoint", autospec=True)
def test_train_no_model(mock_checkpoint):
    """
    Check that training loop runs without crashing, when there is no model
    """
    dummy_net = DummyNet(INPUT_SHAPE, OUTPUT_SHAPE)
    with mock.patch("src.tasks.spectral_u_net.train.SpectralUNet") as net_cls:
        net_cls.return_value = dummy_net
        train(
            num_epochs=2,
            use_cuda=False,
            batch_size=1,
            wandb_name=None,
            subsample=4,
            checkpoint_epochs=None,
        )


@mock.patch("src.utils.trainer.checkpoint", autospec=True)
def test_train_with_model(mock_checkpoint):
    """
    Check that training loop runs without crashing, when there is a model
    """
    train(
        num_epochs=2,
        use_cuda=False,
        batch_size=1,
        wandb_name=None,
        subsample=4,
        checkpoint_epochs=None,
    )