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
    return np.log(clamp(spec, 1e-10))


def log_mel_spec_to_audio(log_spec):
    """
    Estimate audio signal from log-mel spectrogram
    using Griffin-Lim algorithm.
    """
    spec = np.exp(log_spec)
    return mel_to_audio(spec, n_iter=32, **LIBROSA_SPEC_KWARGS)


def clamp(arr, floor):
    """
    Clamps all elements in arr to be larger or equal floor.
    """
    arr[arr < floor] = floor
    return arr


def audio_to_waveglow_spec(audio_arr):
    """
    Convert audio to log-magnitude mel-spectrogram compatible with WaveGlow vocoder.
    """
    mel_spec = melspectrogram(audio_arr, **WAVEGLOW_SPEC_KWARGS)
    return np.log(clamp(mel_spec, 1e-10))


def waveglow_spec_to_audio(spec_t, waveglow):
    assert len(spec_t.shape) == 3, "Incorrect shape"
    assert spec_t.shape[0] == 1, "Only one sample at a time"
    assert spec_t.shape[1] == WAVEGLOW_BINS, "Incorrect number of bins"
    with torch.no_grad():
        audio_t = waveglow.infer(spec_t)

    return audio_t[0].data.cpu().numpy()


def load_waveglow(use_cuda=False):
    """
    Load WaveGlow vocoder model
    https://github.com/NVIDIA/waveglow
    """
    waveglow = torch.hub.load(WAVEGLOW_GITHUB, WAVEGLOW_MODEL_NAME)
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.cuda() if use_cuda else waveglow.cpu()
    return waveglow.eval()


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

# Setup FFT to be compatible with the NVIDIA WaveGlow vocoder implementation.
# https://github.com/NVIDIA/waveglow
WAVEGLOW_GITHUB = "nvidia/DeepLearningExamples:torchhub"
WAVEGLOW_MODEL_NAME = "nvidia_waveglow"
WAVEGLOW_BINS = 80
WAVEGLOW_N_FFT = 1024  # timesteps
WAVEGLOW_HOP = 256  # timesteps
WAVEGLOW_WINDOW = 1024  # timesteps
WAVEGLOW_SAMPLING_RATE = 22050
WAVEGLOW_SPEC_KWARGS = {
    "sr": SAMPLING_RATE,
    "n_mels": WAVEGLOW_BINS,
    "n_fft": WAVEGLOW_N_FFT * SAMPLING_RATE // WAVEGLOW_SAMPLING_RATE,
    "hop_length": WAVEGLOW_HOP * SAMPLING_RATE // WAVEGLOW_SAMPLING_RATE,
    "win_length": WAVEGLOW_WINDOW * SAMPLING_RATE // WAVEGLOW_SAMPLING_RATE,
    "window": "hann",
    "center": True,
    "pad_mode": "reflect",
    "power": 1,
}
