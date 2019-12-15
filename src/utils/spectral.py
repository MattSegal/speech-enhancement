import torch
import numpy as np
from scipy import signal

SAMPLING_RATE = 16000
HOP_MS = 16
WIN_MS = 32  # TODO: Try 64ms


def audio_to_spec(audio_arr, window_ms=WIN_MS, hop_ms=HOP_MS):
    """
    Converts an audio array into linear spectrogram using Hann window.
    returns an array (2, 257, 129) with shape
    channels x freqs x time
    channel 0 is frequency magnitude
    channel 1 is phase
    """
    num_segment = ms_to_steps(window_ms)
    num_overlap = num_segment - ms_to_steps(hop_ms)
    _, _, spectral_frames = signal.stft(
        audio_arr, fs=SAMPLING_RATE, nperseg=num_segment, noverlap=num_overlap,
    )
    mag_frames = spectral_frames.real
    phase_frames = spectral_frames.imag
    return np.stack([mag_frames, phase_frames])


def spec_to_audio(spectral_frames, window_ms=WIN_MS, hop_ms=HOP_MS):
    """
    Reverse audio_to_spec
    """
    mag, phase = spectral_frames
    spectral_frames = mag + 1j * phase
    num_segment = ms_to_steps(window_ms)
    num_overlap = num_segment - ms_to_steps(hop_ms)
    _, audio_arr = signal.istft(
        spectral_frames, fs=SAMPLING_RATE, nperseg=num_segment, noverlap=num_overlap,
    )
    return audio_arr


def preprocess_input_spec(input_t):
    return normalise(logify(input_t))


def logify(arr):
    log_arr = torch.zeros(arr.shape, device=arr.device)
    log_arr[arr > 0] = torch.log(arr[arr > 0])
    log_arr[arr < 0] = -1 * torch.log(-1 * arr[arr < 0])
    return log_arr


def normalise(arr):
    mean = arr.mean()
    centred_arr = arr - mean
    arr_max = torch.abs(centred_arr).max()
    return centred_arr / arr_max


def ms_to_steps(ms):
    """
    Turn time in miliseconds into discrete audio time steps 
    """
    return int((1e-3 * ms) * SAMPLING_RATE)

