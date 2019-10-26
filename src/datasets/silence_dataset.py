import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

NUM_SAMPLES = 500
SAMPLE_LENGTH = 32767


class SilenceDataset(Dataset):
    """
    A dataset of silence
    """

    def __init__(self, train):
        dataset_label = "training" if train else "validation"
        print(f"Loading {dataset_label} silence dataset into memory.")

        print("Loading clean data...")
        self.clean_data = [
            np.zeros(SAMPLE_LENGTH).astype("float32") for _ in range(NUM_SAMPLES)
        ]

        print("Loading noisy data...")
        self.noisy_data = [
            np.zeros(SAMPLE_LENGTH).astype("float32") for _ in range(NUM_SAMPLES)
        ]

        print("Done loading dataset into memory.")

    def __len__(self):
        """
        How many samples there are in the dataset.
        """
        return len(self.noisy_data)

    def __getitem__(self, idx):
        """
        Get item by integer index,
        """
        return torch.tensor(self.noisy_data[idx]), torch.tensor(self.clean_data[idx])
