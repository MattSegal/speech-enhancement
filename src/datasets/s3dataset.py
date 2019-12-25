import os

from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.io import wavfile

from src.utils import s3


class S3BackedDataset(Dataset):
    SAMPLING_RATE = 16000

    def __init__(self, dataset_name, quiet=True):
        self.dataset_name = dataset_name
        self.data_path = os.path.join("data", self.dataset_name)
        self.quiet = quiet
        self.load_s3_data()

    def load_s3_data(self):
        if not os.path.exists(self.data_path):
            if not self.quiet:
                print(f"Fetching {self.dataset_name} data from S3")

            os.makedirs(self.data_path, exist_ok=True)
            s3.fetch_data(self.dataset_name, self.data_path, quiet=self.quiet)
            if not self.quiet:
                print(f"Done fetching {self.dataset_name} data from S3")

    def upload_s3_data(self):
        print(f"Uploading {self.dataset_name} data to S3")
        s3.upload_data(self.data_path, self.dataset_name)

    def find_wav_filenames(self, folder, subsample=None):
        return self._find_ext_filenames(".wav", folder, subsample=subsample)

    def find_flac_filenames(self, folder, subsample=None):
        return self._find_ext_filenames(".flac", folder, subsample=subsample)

    def _find_ext_filenames(self, ext, folder, subsample=None):
        filenames = os.listdir(folder)
        ext_filenames = [f for f in filenames if f.endswith(ext)]
        if subsample:
            ext_filenames = ext_filenames[:subsample]

        return ext_filenames

    def load_data(self, filenames, folder, data):
        """
        Load .wav files into data array.
        """
        itr = list if self.quiet else tqdm
        for filename in itr(filenames):
            path = os.path.join(folder, filename)
            sample_rate, wav_arr = wavfile.read(path)
            assert len(wav_arr.shape) == 1
            assert sample_rate == self.SAMPLING_RATE
            data.append(wav_arr)

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()
