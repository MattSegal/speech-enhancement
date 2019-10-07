import os
import random

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

DATA_PATH = "data/chime"
USED_LABELS = ["v", "c", "f", "m", "b", "p", "o", "U"]


class ChimeDataset(Dataset):
    """
    Loads data from the CHiME Home dataset.
    https://core.ac.uk/download/pdf/30342817.pdf

    Data is based on 6.8 hours of domestic environment audio recordings.
    Each recording is 4 seconds long, and is annotated with one *or more* labels,
    which denote an audio event that occurs with in the recording.

    Sound labels

    v Video game/TV
    c Child speech
    f Adult female speech
    m Adult male speech
    b Broadband noise, e.g. household appliances
    p Percussive sounds, e.g. crash, bang, knock, footsteps
    o Other identifiable sounds
    S Silence / background noise only
    U Flag chunk (unidentifiable sounds, not sure how to label)
    """

    labels = USED_LABELS

    def __init__(self, train):
        """
        """
        self.train = train
        dataset_label = "development" if train else "evaluation"
        print(f"\nLoading CHiME {dataset_label} dataset into memory.")
        csv_path = os.path.join(DATA_PATH, f"{dataset_label}_chunks_refined.csv")
        audio_dir = os.path.join(DATA_PATH, "audio")
        labels_dir = os.path.join(DATA_PATH, "labels")

        # Map idx / labels
        self.idx_to_label = {}
        self.label_to_idx = {}
        for idx, label in enumerate(self.labels):
            self.idx_to_label[idx] = label
            self.label_to_idx[label] = idx

        # Get dataset filenames
        dataset_filenames = read_dataset_filenames(csv_path)

        # Read audio and label info, for each file in the dataset
        self.data = []
        self.data_labels = []
        for filename in tqdm(dataset_filenames):
            audio_path = os.path.join(audio_dir, f"{filename}.wav")
            label_path = os.path.join(labels_dir, f"{filename}.csv")
            labels_arr = read_label_file(label_path)
            audio_arr = read_audio_file(audio_path)
            self.data.append(audio_arr)
            self.data_labels.append(labels_arr)

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
        labels_arr = self.data_labels[idx]
        return torch.tensor(input_arr), torch.tensor(labels_arr)


def read_label_file(label_path):
    """
    Read the sample's label from file.
    Construct 1-encoded array indicating whether label is present in sample.
    Eg "vf" => np.array([1, 0, 1, 0, 0, 0, 0, 0])
    """
    with open(label_path, "r") as f:
        label_text = f.read()

    for line in label_text.split("\n"):
        if line.startswith("majorityvote"):
            label_str = line.split(",")[-1]

    labels_arr = np.zeros(len(USED_LABELS))
    for i in range(len(USED_LABELS)):
        if USED_LABELS[i] in label_str:
            labels_arr[i] = 1

    return labels_arr.astype("float32")


def read_audio_file(audio_path):
    """
    Read the sample's audio
    """
    sample_rate, wav_arr = wavfile.read(audio_path)
    assert sample_rate == 16000
    # The audio file should be mono.
    assert len(wav_arr.shape) == 1, "Audio data should be mono"
    assert wav_arr.shape == (64000,)
    return wav_arr


def read_dataset_filenames(csv_path):
    with open(csv_path, "r") as f:
        dataset_files_text = f.read()

    return [line.split(",")[-1] for line in dataset_files_text.split("\n") if line]
