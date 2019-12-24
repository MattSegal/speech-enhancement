import torch

from src.tasks.acoustic_scenes_spectral.model import SpectralSceneNet, NUM_LABELS

USE_CUDA = torch.cuda.is_available()


def _cuda_maybe(torchy):
    return torchy.cuda() if USE_CUDA else torchy.cpu()


def _get_net(*args, **kwargs):
    return _cuda_maybe(SpectralSceneNet(*args, **kwargs))


def _get_noise(shape):
    return _cuda_maybe(torch.zeros(shape).detach().uniform_())


def test_processes_noise():
    net = _get_net()
    input_shape = (1, 1, 80, 256)
    inputs = _get_noise(input_shape)
    outputs = net(inputs)
    assert outputs.shape == (1, NUM_LABELS)


def test_processes_noise_with_feature_layers():
    net = _get_net().set_feature_mode(num_layers=6)
    input_shape = (1, 1, 80, 256)
    inputs = _get_noise(input_shape)
    outputs = net(inputs)
    assert not outputs
    assert len(net.feature_layers) == 6
    assert net.feature_layers[0].shape == (1, 64, 80, 256)
    assert net.feature_layers[5].shape == (1, 256, 20, 64)
