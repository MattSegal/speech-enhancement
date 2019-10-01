import requests
import torch
from torch.utils.data import Dataset

DATA_DIR = "data"
BASE_URL = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2791/"
TRAIN_DATA = [
    {"compressed": "clean_trainset_28spk_wav.zip", "target": "training_set/clean"},
    {"compressed": "noisy_trainset_28spk_wav.zip", "target": "training_set/noisy"},
]
VALIDATION_DATA = [
    {"compressed": "clean_testset_wav.zip", "target": "validation_set/clean"},
    {"compressed": "noisy_testset_wav.zip", "target": "validation_set/noisy"},
]


class TextDataset(Dataset):
    def __init__(self, train):
        """
        Argument 'num_chars' determines how many characters are used to predict the next character.
        """
        data_meta = TRAIN_DATA if train else VALIDATION_DATA
        for meta_item in data_meta:
            if not os.path.exists(meta_item["target"]):
                print(f"Downloading data...", end=" ")
                url = f"{BASE_URL}{meta_item["path"]}"
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(TARGET_FILE, "wb") as f:
                    for block in response.iter_content(1024):
                        f.write(block)

            print("done")

    def load_text(self):
        """
        Load the text from disk and convert characters to integers with a mapping.
        """
        with open(TARGET_FILE) as f:
            self.text = f.read().lower()

        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

    def __len__(self):
        """
        How many samples there are in the dataset.
        """
        return len(self.text) - self.num_chars - 1

    def __getitem__(self, idx):
        """
        Get item by integer index,
        returns (x_1, x_2, ..., x_n), y
        where x 1 to n are integers representing the input characters
        and y is an integer the output character
        """
        inputs = torch.tensor(
            [self.char_to_idx[c] for c in self.text[idx : idx + self.num_chars]]
        )
        label = self.char_to_idx[self.text[idx + self.num_chars]]
        return inputs, label
