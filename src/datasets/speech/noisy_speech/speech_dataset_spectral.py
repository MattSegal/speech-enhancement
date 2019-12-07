import numpy as np
import torch

from src.utils import spectral

from .speech_dataset import NoisySpeechDataset


class NoisySpectralSpeechDataset(NoisySpeechDataset):
    """
    Get item by integer index,
    """

    def __getitem__(self, idx):
        """
        Returns noisy and clean linear spectrograms, inc. phase info
        Throw away some time / spectral info to make shaped dims fit into power of 2s.
        """
        noisy_spectral = spectral.audio_to_spec(self.noisy_data[idx])
        noisy_spectral = noisy_spectral[:, :-1, :-1]
        clean_spectral = spectral.audio_to_spec(self.clean_data[idx])
        clean_spectral = clean_spectral[:, :-1, :-1]
        return torch.tensor(noisy_spectral), torch.tensor(clean_spectral)
