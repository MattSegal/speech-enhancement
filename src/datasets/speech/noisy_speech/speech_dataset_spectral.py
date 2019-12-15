import numpy as np
import torch

from src.utils import spectral

from .speech_dataset import NoisySpeechDataset

MAX_AUDIO_LENGTH = 2 ** 16  # ~4s of data at 16kHz


class NoisySpectralSpeechDataset(NoisySpeechDataset):
    """
    Get item by integer index,
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling_factor = 10  # Chosen by eyeballing data

    def normalize(self, arr):
        """
        Push values into +/- 1 range
        """
        return arr * self.scaling_factor

    def denormalize(self, arr):
        return arr / self.scaling_factor

    def __getitem__(self, idx):
        """
        Returns noisy and clean linear spectrograms, inc. phase info
        Throw away some time / spectral info to make shaped dims fit into power of 2s.
        """
        noisy_spectral = spectral.audio_to_spec(self.noisy_data[idx])
        noisy_spectral = self.normalize(noisy_spectral[:, :-1, :-1])
        clean_spectral = spectral.audio_to_spec(self.clean_data[idx])
        clean_spectral = self.normalize(clean_spectral[:, :-1, :-1])
        return torch.tensor(noisy_spectral), torch.tensor(clean_spectral)
