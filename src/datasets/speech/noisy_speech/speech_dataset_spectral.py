import torch
import numpy as np

from src.utils import spectral

from .speech_dataset import NoisySpeechDataset

MAX_AUDIO_LENGTH = 47360  # ~3s of data at 16kHz


class NoisySpectralSpeechDataset(NoisySpeechDataset):
    """
    Get item by integer index,
    """

    MAX_AUDIO_LENGTH = MAX_AUDIO_LENGTH

    def process_sample(self, sample_arr):
        # Convert audio array to mel spectrum
        sample_spec = spectral.audio_to_waveglow_spec(sample_arr)
        assert sample_spec.shape == (80, 256)
        # Add channel dimension
        sample_spec = np.expand_dims(sample_spec, axis=0)
        return torch.tensor(sample_spec)

    def __getitem__(self, idx):
        """
        Returns noisy and clean log magnitude spectrograms
        """
        noisy_spectral = self.process_sample(self.noisy_data[idx])
        clean_spectral = self.process_sample(self.clean_data[idx])
        return noisy_spectral, clean_spectral
