import os

from tqdm import tqdm
from scipy.io import wavfile

SAVE_DIR = "data/saver"


def save_data(dataset, num_samples, name):
    """
    Saves some of an audio dataset to disk
    """
    data_dir = os.path.join(SAVE_DIR, name)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Saving dataset to {data_dir}.")
    for idx in tqdm(range(num_samples)):
        if idx + 1 > num_samples:
            break

        noisy, clean = dataset[idx]
        noisy_path = os.path.join(data_dir, f"{idx}_noisy.wav")
        clean_path = os.path.join(data_dir, f"{idx}_clean.wav")
        wavfile.write(noisy_path, 16000, noisy.detach().numpy())
        wavfile.write(clean_path, 16000, clean.detach().numpy())
