from src.datasets import SpectralSceneDataset as Dataset


def test_get_item():
    dataset = Dataset(train=True, subsample=8, quiet=True)
    input_spec, label_idx = dataset[0]
    assert input_spec.shape == (1, 80, 256)
    assert type(label_idx) is int
