import torch

from src.utils import augment
from src.datasets.noisy_speech.speech_dataset import NoisySpeechDataset


class AugmentedSpeechDataset(NoisySpeechDataset):
    def __getitem__(self, idx):
        """
        Get item by integer index,
        """
        clean = self.clean_data[idx]
        noisy = augment.mask_high_freq(clean, mask_freq=3000).astype("float32")
        return torch.tensor(noisy), torch.tensor(clean)
