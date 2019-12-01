import numpy as np
from scipy import signal

SAMPLING_RATE = 16000
SAMPLES_PER_SEGMENT = 512


def audio_to_spec(audio_arr):
    """
    Converts an audio array into linear spectrogram
    returns an array (2, 257, 129) with shape
    channels x freqs x time
    channel 0 is frequency magnitude
    channel 1 is phase
    """
    _, _, spectral_frames = signal.stft(audio_arr, fs=SAMPLING_RATE, nperseg=SAMPLES_PER_SEGMENT)
    mag_frames = spectral_frames.real
    phase_frames = spectral_frames.imag
    return np.stack([mag_frames, phase_frames])


def spec_to_audio(spectral_frames):
    """
    Reverse audio_to_spec
    """
    mag, phase = spectral_frames
    spectral_frames = mag + 1j * phase
    _, audio_arr = signal.istft(res_rec, fs=SAMPLING_RATE, nperseg=SAMPLES_PER_SEGMENT)
    return audio_arr

