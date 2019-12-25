import os
import random

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

from src.datasets.s3dataset import S3BackedDataset

DATASET_NAME = "scenes"
DATA_PATH = "data/"
SAMPLING_RATE = 16000
CHUNK_SIZE = 32767
CLASS_LABELS = [
    "bus",
    "car",
    "forest_path",
    "cafe/restaurant",
    "residential_area",
    "library",
    "park",
    "grocery_store",
    "city_center",
    "beach",
    "train",
    "tram",
    "home",
    "office",
    "metro_station",
]


class SceneDataset(S3BackedDataset):
    """
    TUT acoustic scenes dataset.

    A dataset of acoustic scenes and their label, for use in the acoustic scene classification task.
    The input is a 1D tensor of floats, representing a complete noisy audio sample.
    The target is an integer, representing a scene label. 

    http://www.cs.tut.fi/sgn/arg/dcase2017/challenge/task-acoustic-scene-classification
    """

    labels = CLASS_LABELS
    CHUNK_SIZE = CHUNK_SIZE

    def __init__(self, train, subsample=None, quiet=True):
        """
        Load the dataset into memory so it can be used for training.
        """
        super().__init__(dataset_name=DATASET_NAME, quiet=quiet)
        dataset_label = "train" if train else "test"
        data_folder = os.path.join(DATA_PATH, "scenes", f"{dataset_label}_set")
        print(f"\nLoading TUT {dataset_label} dataset into memory.")

        # Load class labels from a text file.
        print("Loading class labels...")
        meta_path = os.path.join(data_folder, "meta.txt")
        with open(meta_path, "r") as f:
            meta_text = f.read()

        label_lookup = {}
        for line in meta_text.split("\n"):
            if line:
                filename, label = line.split("\t")
                assert label in CLASS_LABELS
                filename_cleaned = filename.replace("audio/", "")
                label_lookup[filename_cleaned] = label

        self.idx_to_label = {}
        self.label_to_idx = {}
        for idx, label in enumerate(CLASS_LABELS):
            self.idx_to_label[idx] = label
            self.label_to_idx[label] = idx

        # Load audio data from .wav files, associate each file with its label.
        print("Loading data...")
        self.data = []
        self.data_labels = []
        wav_files = [
            filename for filename in os.listdir(data_folder) if filename.endswith(".wav")
        ]

        if subsample:
            wav_files = wav_files[:subsample]

        for filename in tqdm(wav_files):
            # Get the label for this file
            label = label_lookup[filename]
            label_idx = self.label_to_idx[label]
            # Read the audio file into memory
            path = os.path.join(data_folder, filename)
            sample_rate, wav_arr = wavfile.read(path)
            assert sample_rate == SAMPLING_RATE
            # The audio files are stereo: split them into two mono files.
            assert len(wav_arr.shape) == 2, "Audio data should be stereo"
            wav_arr = wav_arr.transpose()
            mono_wav_arrs = (wav_arr[0], wav_arr[1])

            # Split each file up into non-overlapping chunks
            for wav_arr in mono_wav_arrs:
                # Add each audio segment to the dataset
                chunks = split_even_chunks(wav_arr, self.CHUNK_SIZE)
                for chunk in chunks:
                    chunk_contiguous = np.asfortranarray(chunk)
                    self.data.append(chunk_contiguous)
                    self.data_labels.append(label_idx)

        assert len(self.data) == len(self.data_labels)
        print(f"Done loading dataset into memory: loaded {len(self.data)} items.\n")

    def __len__(self):
        """
        How many samples there are in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get item by integer index,
        returns input_t, label_idx
            input: (CHUNK_SIZE, )
            label: integer
        """
        input_arr = self.data[idx]
        label_idx = self.data_labels[idx]
        return torch.tensor(input_arr), label_idx


def split_even_chunks(input_arr, chunk_size):
    """
    Split the audio sample into multiple even chunks,
    with a random offset.
    """
    even_length = len(input_arr) - len(input_arr) % chunk_size
    remainder_length = len(input_arr) - even_length
    offset = np.random.randint(0, remainder_length + 1)
    num_chunks = even_length / chunk_size
    start = offset
    end = offset + even_length
    chunks = np.split(input_arr[start:end], num_chunks)
    return chunks
