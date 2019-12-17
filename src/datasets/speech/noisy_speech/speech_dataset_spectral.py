import numpy as np
import torch
from librosa.util import normalize

from src.utils import spectral

from .speech_dataset import NoisySpeechDataset

MAX_AUDIO_LENGTH = 2 ** 16  # ~4s of data at 16kHz


class NoisySpectralSpeechDataset(NoisySpeechDataset):
    """
    Get item by integer index,
    """

    def process_sample(self, sample_arr, norm):
        # Convert audio array to mel spectrum
        sample_spec = spectral.audio_to_log_mel_spec(sample_arr)
        # Knock off last time step so we have all dims as powers of 2
        sample_spec = sample_spec[:, :-1]
        if norm:
            sample_spec = normalize(sample_spec)

        # Add channel dimension
        sample_spec = np.expand_dims(sample_spec, axis=0)
        return torch.tensor(sample_spec)

    def __getitem__(self, idx):
        """
        Returns noisy and clean log magnitude spectrograms
        """
        noisy_spectral = self.process_sample(self.noisy_data[idx], norm=True)
        clean_spectral = self.process_sample(self.clean_data[idx], norm=False)
        return noisy_spectral, clean_spectral
