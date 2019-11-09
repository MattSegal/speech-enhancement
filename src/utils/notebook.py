"""
iPython notebook tools
"""
import os
import subprocess
from urllib.parse import urlparse

import requests
import torch
import numpy as np
from IPython.display import Audio
import matplotlib.pyplot as plt
from scipy.io import wavfile


class Sampler:
    def __init__(self, net, dataset):
        self.net = net
        self.dataset = dataset

    def get_results(self, idx, loops=1):
        """
        Get results from a sample in the dataset
        """
        noisy_arr = self.dataset[idx][0].numpy()
        clean_arr = self.dataset[idx][1].numpy()
        with torch.no_grad():
            inputs = torch.tensor(noisy_arr).float().cpu()
            inputs = inputs.view(1, 1, -1)
            outputs = inputs
            for _ in range(loops):
                outputs = self.net(outputs)

            outputs = outputs.squeeze(dim=0).squeeze(dim=0)
            pred_clean = outputs.cpu().detach().numpy()
            pred_noise = noisy_arr - pred_clean
            true_noise = noisy_arr - clean_arr
            return noisy_arr, clean_arr, pred_clean, pred_noise, true_noise

    def get_results_from_file(self, file_path, start=0, width=2 ** 18):
        sample_rate, noisy_arr = wavfile.read(file_path)
        if len(noisy_arr.shape) > 1:
            noisy_arr = np.mean(noisy_arr, axis=1)

        assert len(noisy_arr.shape) == 1
        assert sample_rate == 16000
        inputs = noisy_arr[start : width + start]
        with torch.no_grad():
            pred_clean = self.get_pred_clean(inputs)

        pred_noise = inputs - pred_clean
        return inputs, pred_clean, pred_noise

    def get_results_from_mp3_url(self, url, start=0, width=2 ** 18):
        os.makedirs("data/downloads", exist_ok=True)
        url_path = urlparse(url).path
        filename = os.path.basename(url_path)
        file_path = os.path.join("data/downloads", filename)
        wav_file_path = file_path.replace(".mp3", ".wav")
        if not os.path.exists(wav_file_path):
            with open(file_path, "wb") as f:
                resp = requests.get(url, allow_redirects=True)
                f.write(resp.content)

            cmd = f"sox {file_path} -r 16000 -b 32 -e float {wav_file_path}"
            subprocess.run(args=[cmd], shell=True, check=True)

        return self.get_results_from_file(wav_file_path, start, width)

    def get_pred_clean(self, noisy_arr):
        inputs = torch.tensor(noisy_arr).float().cpu()
        inputs = inputs.view(1, 1, -1)
        return self.net(inputs).squeeze(dim=0).squeeze(dim=0).cpu().detach().numpy()


def visualize_audio(arr, print_str):
    """
    Visualize audio results and a waveform, spectrogram and audio file.
    """
    print(print_str)
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    fig.set_size_inches(12, 12)
    ax1.plot(arr)
    ax2.specgram(arr, Fs=16000)
    plt.show()
    return Audio(arr, rate=16000)
