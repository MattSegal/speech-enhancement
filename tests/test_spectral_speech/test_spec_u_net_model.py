import torch

from src.tasks.spectral_u_net.model import SpectralUNet

USE_CUDA = torch.cuda.is_available()


def _cuda_maybe(torchy):
    return torchy.cuda() if USE_CUDA else torchy.cpu()


def _get_net(*args, **kwargs):
    return _cuda_maybe(SpectralUNet(*args, **kwargs))


def _get_noise(shape):
    return _cuda_maybe(torch.zeros(shape).detach().uniform_())


def test_processes_noise():
    net = _get_net()
    input_shape = (1, 1, 80, 256)
    inputs = _get_noise(input_shape)
    outputs = net(inputs)
    assert outputs.shape == (1, 1, 80, 256)
