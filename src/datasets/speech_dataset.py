import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

DATA_PATH = "data/"


class SpeechDataset(Dataset):
    """
    A dataset of clean and noisy speech, for use in the speech enhancement task.
    The input is a 1D tensor of floats, representing a complete noisy audio sample.
    The target is a 1D tensor of floats, representing a corresponding clean audio sample. 
    """

    def __init__(self, train):
        dataset_label = "training" if train else "validation"
        print(f"Loading {dataset_label} dataset into memory.")

        print("Loading clean data...")
        self.clean_data = [torch.tensor(np.random.randn(32767)).float() for _ in range(10)]
        # self.clean_data = []
        # self.clean_folder = os.path.join(DATA_PATH, f"{dataset_label}_set_clean")
        # self.clean_files = os.listdir(self.clean_folder)
        # assert all([f.endswith(".wav") for f in self.clean_files])
        # self.load_data(self.clean_files, self.clean_folder, self.clean_data)

        print("Loading noisy data...")
        self.noisy_data = [torch.tensor(np.random.randn(32767)).float() for _ in range(10)]
        # self.noisy_data = []
        # self.noisy_folder = os.path.join(DATA_PATH, f"{dataset_label}_set_noisy")
        # self.noisy_files = os.listdir(self.noisy_folder)
        # assert all([f.endswith(".wav") for f in self.noisy_files])
        # assert len(self.noisy_files) == len(self.clean_files)
        # self.load_data(self.noisy_files, self.noisy_folder, self.noisy_data)

        print("Done loading dataset into memory.")

    def load_data(self, filenames, folder, data):
        """
        Load .wav files into memory.
        """
        for filename in tqdm(filenames):
            path = os.path.join(folder, filename)
            sample_rate, wav_arr = wavfile.read(path)
            assert len(wav_arr.shape) == 1
            assert sample_rate == 16000
            data.append(torch.tensor(wav_arr))

    def __len__(self):
        """
        How many samples there are in the dataset.
        """
        return len(self.noisy_data)

    def __getitem__(self, idx):
        """
        Get item by integer index,
        """
        return self.noisy_data[idx], self.clean_data[idx]
