import torch

from src.utils.augment import augment_audio
from src.datasets.noisy_speech.speech_dataset import NoisySpeechDataset


class AugmentedSpeechDataset(NoisySpeechDataset):
    def __getitem__(self, idx):
        """
        Get item by integer index,
        """
        clean = self.clean_data[idx]
        noisy = augment_audio(clean).astype("float32")
        return torch.tensor(noisy), torch.tensor(clean)

