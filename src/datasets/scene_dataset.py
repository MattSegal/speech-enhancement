import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

DATA_PATH = "data/"
CHUNK_SIZE = 2 ** 15 - 1


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
        dataset_label = "training" if train else "validation"
        print(f"Loading {dataset_label} dataset into memory.")
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

        # HACK - ONLY USE A SUBSET OF FILES
        wav_files = wav_files[:100]

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
                even_length = len(wav_arr) - len(wav_arr) % CHUNK_SIZE
                num_chunks = even_length / CHUNK_SIZE
                chunks = np.split(wav_arr[:even_length], num_chunks)
                for chunk in chunks:
                    # Add each chunk to the dataset
                    self.data.append(torch.tensor(chunk))
                    self.data_labels.append(label_idx)

        assert len(self.data) == len(self.data_labels)
        print(f"Done loading dataset into memory: loaded {len(self.data)} items.")

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
        input_t = self.data[idx]
        label_idx = self.data_labels[idx]
        return input_t, label_idx
