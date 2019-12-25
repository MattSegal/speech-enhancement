import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

from src.utils import s3
from src.datasets.s3dataset import S3BackedDataset

DATASET_NAME = "noisy_speech"
MAX_AUDIO_LENGTH = 2 ** 15  # ~2s of data at 16kHz


class NoisySpeechDataset(S3BackedDataset):
    """
    A dataset of clean and noisy speech, for use in the speech enhancement task.
    The input is a 1D tensor of floats, representing a complete noisy audio sample.
    The target is a 1D tensor of floats, representing a corresponding clean audio sample. 
    """

    MAX_AUDIO_LENGTH = MAX_AUDIO_LENGTH
    SAMPLING_RATE = 16000

    def __init__(self, train, subsample=None, quiet=True):
        self.clean_only = False
        self.quiet = quiet
        super().__init__(dataset_name=DATASET_NAME, quiet=quiet)
        dataset_label = "training" if train else "validation"
        if not quiet:
            print(f"Loading {dataset_label} dataset into memory.")
            print("Loading clean data...")

        self.clean_data = []
        self.clean_folder = os.path.join(self.data_path, f"{dataset_label}_set_clean")
        self.wav_filenames = self.find_wav_filenames(
            self.clean_folder, subsample=subsample
        )
        self.load_and_trim_data(self.wav_filenames, self.clean_folder, self.clean_data)
        if not quiet:
            print("Loading noisy data...")

        self.noisy_data = []
        self.noisy_folder = os.path.join(self.data_path, f"{dataset_label}_set_noisy")
        self.load_and_trim_data(self.wav_filenames, self.noisy_folder, self.noisy_data)
        if not quiet:
            print("Done loading dataset into memory.")

    def load_and_trim_data(self, filenames, folder, data):
        """
        Load .wav files into memory, but trim too long ones..
        """
        self.load_data(filenames, folder, data)

        # Ensure all sound samples are the same length
        for idx, wav_arr in enumerate(data):
            if len(wav_arr) > self.MAX_AUDIO_LENGTH:
                # Shorten audio sample
                data[idx] = subsample_chunk(wav_arr, self.MAX_AUDIO_LENGTH)
            elif len(wav_arr) < self.MAX_AUDIO_LENGTH:
                # Pad sample with zeros
                data[idx] = pad_chunk(wav_arr, self.MAX_AUDIO_LENGTH)

    def __len__(self):
        """
        How many samples there are in the dataset.
        """
        return len(self.noisy_data)

    def __getitem__(self, idx):
        """
        Get item by integer index,
        """
        clean_t = torch.tensor(self.clean_data[idx])
        noisy_t = torch.tensor(self.noisy_data[idx])
        if self.clean_only:
            return clean_t, clean_t
        else:
            return noisy_t, clean_t


def subsample_chunk(input_arr, chunk_width):
    """
    Sample a length of audio, so that it's always
    the same size as all other samples (required for mini-batching)
    """
    assert chunk_width < len(input_arr)
    input_arr = input_arr[:chunk_width]
    return input_arr


def pad_chunk(input_arr, chunk_width):
    """
    Pad sample length of audio, so that it's always
    the same size as all other samples (required for mini-batching)
    """
    assert chunk_width > len(input_arr)
    padding = chunk_width - input_arr.size
    input_arr = np.pad(input_arr, (0, padding))
    return input_arr
