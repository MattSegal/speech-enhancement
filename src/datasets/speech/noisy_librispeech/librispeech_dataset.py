import os
import random

import torchaudio
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from src.utils import s3
from src.datasets.s3dataset import S3BackedDataset

from . import settings

DATASET_NAME = settings.DATASET_NAME
AUDIO_LENGTH = settings.AUDIO_LENGTH


class NoisyLibreSpeechDataset(S3BackedDataset):
    """
    A dataset of clean and noisy speech, for use in the speech enhancement task.
    The input is a 1D tensor of floats, representing a complete noisy audio sample.
    The target is a 1D tensor of floats, representing a corresponding clean audio sample. 
    """

    def __init__(self, noise_data, train, subsample=None, quiet=True):
        self.quiet = quiet
        self.noise_data = noise_data
        super().__init__(dataset_name=DATASET_NAME, quiet=quiet)
        dataset_label = "train" if train else "test"
        itr = list if self.quiet else tqdm
        self.clean_data = []
        self.clean_folder = os.path.join(self.data_path, f"{dataset_label}_set")
        self.clean_filenames = self.find_flac_filenames(
            self.clean_folder, subsample=subsample
        )
        if not quiet:
            print(f"Loading {dataset_label} dataset into memory.")
            print("Loading clean data...")

        for filename in itr(self.clean_filenames):
            path = os.path.join(self.clean_folder, filename)
            tensor, sample_rate = torchaudio.load(path)
            if tensor.nelement() < AUDIO_LENGTH:
                continue

            assert sample_rate == 16000
            assert tensor.dtype == torch.float32
            tensor = tensor.reshape(-1)
            self.clean_data.append(tensor)

        if not quiet:
            print("Done loading dataset into memory.")

    def __len__(self):
        """
        How many samples there are in the dataset.
        """
        return len(self.clean_data)

    def __getitem__(self, idx):
        """
        Get item by integer index,
        """
        clean = self.clean_data[idx]
        noise_idx = random.randint(0, len(self.noise_data) - 1)
        noise = self.noise_data[noise_idx]
        clean_chunk = subsample_chunk_random(clean, AUDIO_LENGTH)
        noise_chunk = subsample_chunk_random(noise, AUDIO_LENGTH)
        noise_chunk = noise_chunk * random.randint(1, 10)
        noise_chunk[noise_chunk > 1] = 1
        noise_chunk[noise_chunk < -1] = -1
        noisy_chunk = clean_chunk + noise_chunk
        return noisy_chunk, clean_chunk


def subsample_chunk_random(tensor, chunk_width):
    """
    Randomly sample length of audio, so that it's always
    the same size as all other samples (required for mini-batching)
    """
    size = tensor.nelement()
    assert chunk_width < size
    chunk_start = random.randint(0, size - chunk_width)
    return tensor[chunk_start : chunk_start + chunk_width]
