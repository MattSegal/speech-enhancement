import torch
import numpy as np

from src.utils import spectral

from .scene_dataset import SceneDataset

CHUNK_SIZE = 47360  # ~3s of data at 16kHz


class SpectralSceneDataset(SceneDataset):
    """
    TUT acoustic scenes dataset, using mel-spectrograms as the input feature
    """

    CHUNK_SIZE = CHUNK_SIZE

    def process_sample(self, sample_arr):
        # Convert audio array to mel spectrum
        sample_spec = spectral.audio_to_waveglow_spec(sample_arr)
        assert sample_spec.shape == (80, 256)
        # Add channel dimension
        sample_spec = np.expand_dims(sample_spec, axis=0)
        return torch.tensor(sample_spec)

    def __getitem__(self, idx):
        """
        Get item by integer index,
        returns input_t, label_idx
            input_spec: (1, 80, 256)
            label: integer
        """
        input_spec = self.process_sample(self.data[idx])
        label_idx = self.data_labels[idx]
        return input_spec, label_idx
