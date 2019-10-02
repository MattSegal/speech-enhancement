import os

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

DATA_PATH = "data/"


class SceneDataset(Dataset):
    """
    A dataset of acoustic scenes and their label, for use in the acoustic scene classification task.
    The input is a 1D tensor of floats, representing a complete noisy audio sample.
    The target is an integer, representing a scene label. 
    """

    def __init__(self, train):
        dataset_label = "training" if train else "validation"
        print(f"Loading {dataset_label} dataset into memory.")
        data_folder = os.path.join(DATA_PATH, f"scenes_{dataset_label}_set")

        print("Loading metadata...")
        meta_path = os.path.join(data_folder, "meta.txt")
        with open(meta_path, "r") as f:
            meta_text = f.read()

        label_lookup = {}
        self.labels = set()
        for line in meta_text.split("\n"):
            if line:
                filename, label = line.split("\t")
                label_lookup[filename] = label
                self.labels.add(label)

        self.idx_to_label = {}
        self.label_to_idx = {}
        for idx, label in enumerate(self.labels):
            self.idx_to_label[idx] = label
            self.label_to_idx[label] = idx

        print("Loading data...")
        self.data = []
        self.data_labels = []
        wav_files = [
            filename
            for filename in os.listdir(data_folder)
            if filename.endswith(".wav")
        ]

        # HACK - ONLY USE 1 FILE
        wav_files = wav_files[:1]

        for filename in tqdm(wav_files):
            path = os.path.join(data_folder, filename)
            sample_rate, wav_arr = wavfile.read(path)
            assert len(wav_arr.shape) == 2, "Audio data should be stereo"
            assert sample_rate == 16000
            wav_arr = wav_arr.sum(axis=1) / 2
            assert len(wav_arr.shape) == 1, "Audio data should now be mono"
            self.data.append(torch.tensor(wav_arr))
            try:
                label = label_lookup[filename]
            except KeyError:
                continue

            label_idx = self.label_to_idx[label]
            self.data_labels.append(label_idx)

        print("Done loading dataset into memory.")

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
