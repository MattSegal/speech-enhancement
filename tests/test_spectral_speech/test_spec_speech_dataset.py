from src.datasets import NoisySpectralSpeechDataset as Dataset


def test_len():
    dataset = Dataset(train=True, subsample=8, quiet=True)
    assert len(dataset) == 8


def test_get_item():
    dataset = Dataset(train=True, subsample=8, quiet=True)
    noisy, clean = dataset[0]
    assert noisy.shape == clean.shape
    assert noisy.shape == (1, 80, 256)
