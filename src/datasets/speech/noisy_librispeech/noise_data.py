import os
import random

import torchaudio
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from src.datasets.s3dataset import S3BackedDataset

from . import settings

DATASET_NAME = settings.DATASET_NAME
AUDIO_LENGTH = settings.AUDIO_LENGTH


class NoisyScenesDataset(S3BackedDataset):
    """
    A dataset of noisy scenes, for use in the speech enhancement task.
    Originally taken from the TUT acoustic scenes dataset.
    """

    def __init__(self, subsample=None, quiet=True):
        self.quiet = quiet
        super().__init__(dataset_name=DATASET_NAME, quiet=quiet)
        itr = list if self.quiet else tqdm
        self.noise_data = []
        self.noise_folder = os.path.join(self.data_path, "noise")
        self.noise_filenames = self.find_flac_filenames(
            self.noise_folder, subsample=subsample
        )
        if not quiet:
            print("Loading noisy data...")

        for filename in itr(self.noise_filenames):
            path = os.path.join(self.noise_folder, filename)
            tensor, sample_rate = torchaudio.load(path)
            if tensor.nelement() < AUDIO_LENGTH:
                continue

            assert sample_rate == 16000
            assert tensor.dtype == torch.float32
            tensor = tensor[0, :].reshape(-1)
            self.noise_data.append(tensor)

    def __len__(self):
        """
        How many samples there are in the dataset.
        """
        return len(self.noise_data)

    def __getitem__(self, idx):
        """
        Get item by integer index,
        """
        return self.noise_data[idx]
