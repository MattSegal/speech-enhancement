import torch
import numpy as np
from scipy import signal
from librosa.feature import melspectrogram
from librosa.feature.inverse import mel_to_audio

SAMPLING_RATE = 16000
HOP_MS = 16
WIN_MS = 64


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


def ms_to_steps(ms):
    """
    Turn time in miliseconds into discrete audio time steps 
    """
    return int((1e-3 * ms) * SAMPLING_RATE)


def audio_to_log_mel_spec(audio_arr):
    """
    Get mel-filtered power spectrogram from audio signal. 
    """
    spec = melspectrogram(audio_arr, n_mels=4 * WIN_MS, **LIBROSA_SPEC_KWARGS)
    return np.log(spec + 1e-10)


def log_mel_spec_to_audio(log_spec):
    """
    Estimate audio signal from log-mel spectrogram
    using Griffin-Lim algorithm.
    """
    spec = np.exp(log_spec)
    return mel_to_audio(spec, n_iter=32, **LIBROSA_SPEC_KWARGS)


LIBROSA_SPEC_KWARGS = {
    "sr": SAMPLING_RATE,
    "n_fft": ms_to_steps(WIN_MS),
    "hop_length": ms_to_steps(HOP_MS),
    "win_length": ms_to_steps(WIN_MS),
    "window": "hann",
    "center": True,
    "pad_mode": "reflect",
    "power": 2.0,
}

