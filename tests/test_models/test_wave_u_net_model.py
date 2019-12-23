import torch

from src.tasks.waveunet.models.wave_u_net import WaveUNet

USE_CUDA = torch.cuda.is_available()


def _cuda_maybe(torchy):
    return torchy.cuda() if USE_CUDA else torchy.cpu()


def _get_net(*args, **kwargs):
    return _cuda_maybe(WaveUNet(*args, **kwargs))


def _get_noise(shape):
    return _cuda_maybe(torch.zeros(shape).detach().uniform_())


def test_processes_noise():
    net = _get_net()
    assert net.skips_enabled == True
    input_shape = (1, 2 ** 15)
    inputs = _get_noise(input_shape)
    outputs = net(inputs)
    assert outputs.shape == (1, 2 ** 15)


def test_disable_skips():
    net = _get_net()
    net.skips_enabled = False
    input_shape = (1, 2 ** 15)
    inputs = _get_noise(input_shape)
    outputs = net(inputs)


def test_freeze_entire_model():
    net = _get_net()
    num_params = sum(1 for p in net.parameters())
    num_frozen = sum(1 for p in net.parameters() if p.requires_grad == False)
    assert num_frozen == 0
    net.freeze()
    num_frozen = sum(1 for p in net.parameters() if p.requires_grad == False)
    assert num_frozen == num_params
    net.unfreeze()
    num_frozen = sum(1 for p in net.parameters() if p.requires_grad == False)
    assert num_frozen == 0


def test_freeze_encoder_model():
    net = _get_net()
    num_params = sum(1 for p in net.parameters())
    num_encoder_params = sum(1 for p in net.encoders.parameters())
    assert num_encoder_params < num_params
    num_frozen = sum(1 for p in net.parameters() if p.requires_grad == False)
    assert num_frozen == 0
    net.freeze_encoder()
    num_frozen = sum(1 for p in net.parameters() if p.requires_grad == False)
    assert num_frozen == num_encoder_params

