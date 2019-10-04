import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

DATA_PATH = "data/"
NUM_CONV_LAYERS = 14
MIN_CHUNK_SIZE = 2 ** (NUM_CONV_LAYERS + 1) - 1
MAX_CHUNK_SIZE = 2 ** (NUM_CONV_LAYERS + 2) - 1
SUB_SAMPLE = False


class SceneDataset(Dataset):
    """
    A dataset of acoustic scenes and their label, for use in the acoustic scene classification task.
    The input is a 1D tensor of floats, representing a complete noisy audio sample.
    The target is an integer, representing a scene label. 
    """

    def __init__(self, train):
        """
        Load the dataset into memory so it can be used for training.
        """
        self.train = train
        dataset_label = "training" if train else "validation"
        print(f"\nLoading {dataset_label} dataset into memory.")
        data_folder = os.path.join(DATA_PATH, f"scenes_{dataset_label}_set")

        # Load class labels from a text file.
        print("Loading class labels...")
        meta_path = os.path.join(data_folder, "meta.txt")
        with open(meta_path, "r") as f:
            meta_text = f.read()

        label_lookup = {}
        self.labels = set()
        for line in meta_text.split("\n"):
            if line:
                filename, label = line.split("\t")
                filename_cleaned = filename.replace("audio/", "")
                label_lookup[filename_cleaned] = label
                self.labels.add(label)

        self.idx_to_label = {}
        self.label_to_idx = {}
        for idx, label in enumerate(self.labels):
            self.idx_to_label[idx] = label
            self.label_to_idx[label] = idx

        # Load audio data from .wav files, associate each file with its label.
        print("Loading data...")
        self.data = []
        self.data_labels = []
        wav_files = [
            filename
            for filename in os.listdir(data_folder)
            if filename.endswith(".wav")
        ]

        for filename in tqdm(wav_files):
            # Get the label for this file
            label = label_lookup[filename]
            label_idx = self.label_to_idx[label]
            # Read the audio file into memory
            path = os.path.join(data_folder, filename)
            sample_rate, wav_arr = wavfile.read(path)
            assert sample_rate == 16000
            # The audio files are stereo: split them into two mono files.
            assert len(wav_arr.shape) == 2, "Audio data should be stereo"
            wav_arr = wav_arr.transpose()
            mono_wav_arrs = (wav_arr[0], wav_arr[1])

            # Split each file up into non-overlapping chunks
            for wav_arr in mono_wav_arrs:
                # Add each audio segment to the dataset
                chunks = split_even_chunks(wav_arr)
                for chunk in chunks:
                    self.data.append(chunk)
                    self.data_labels.append(label_idx)

        if SUB_SAMPLE:
            self.data = self.data[:SUB_SAMPLE]
            self.data_labels = self.data_labels[:SUB_SAMPLE]

        assert len(self.data) == len(self.data_labels)
        print(f"Done loading dataset into memory: loaded {len(self.data)} items.\n")

    def __len__(self):
        """
        How many samples there are in the dataset.
        """
        return len(self.data)  # Untrue?

    def __getitem__(self, idx):
        """
        Get item by integer index,
        returns input_t, label_idx
            input:
            label: 
        """
        input_arr = self.data[idx]
        label_idx = self.data_labels[idx]
        return torch.tensor(input_arr), label_idx


def sample_random_chunk(input_arr):
    """
    Randomly sample length of audio then pad it with zeros, so that it's always
    the same size as all other samples (required for mini-batching)
    """
    # Determine chunk width
    random_exponent = np.random.uniform(
        np.log10(MIN_CHUNK_SIZE - 0.5), np.log10(MAX_CHUNK_SIZE + 0.5)
    )
    chunk_width = int(np.round(10.0 ** random_exponent))
    chunk_start = np.random.randint(0, np.size(input_arr) - chunk_width + 1)
    # Extract chunk from input
    input_arr = input_arr[chunk_start : chunk_start + chunk_width]
    # Pad the chunk with zeros to be a uniform length.
    padding = MAX_CHUNK_SIZE - input_arr.size
    input_arr = np.pad(input_arr, (0, padding))
    return input_arr


def split_even_chunks(input_arr):
    """
    Split the audio sample into multiple even chunks
    """
    even_length = len(input_arr) - len(input_arr) % MIN_CHUNK_SIZE
    num_chunks = even_length / MIN_CHUNK_SIZE
    chunks = np.split(input_arr[:even_length], num_chunks)
    return chunks

