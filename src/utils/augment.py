import random

import numpy as np
from scipy import signal

SAMPLING_FREQ = 16000

BAND = "BAND"
LOW = "LOW"
HIGH = "HIGH"
NONE = "NONE"
FREQ_MASKS = [BAND, LOW, HIGH, NONE]


def augment_audio(input_arr):
    mask = random.choice(FREQ_MASKS)
    if mask == BAND:
        aug_arr = mask_freq_band(input_arr)
    elif mask == LOW:
        aug_arr = mask_low_freq(input_arr)
    elif mask == HIGH:
        aug_arr = mask_high_freq(input_arr)
    else:
        aug_arr = input_arr

    if random.random() > 0.5:
        aug_arr = add_noise(aug_arr)

    return aug_arr


def add_noise(input_arr, noise_stdev=None):
    """
    Add gaussian noise to input arr
    """
    noise_stdev = noise_stdev if noise_stdev else random.uniform(1e-4, 1e-3)
    return input_arr + np.random.normal(0, noise_stdev, input_arr.size)


def mask_freq_band(input_arr, mask_size=None, mask_start=None):
    """
    Randomly removes freqency band
    from input array using band stop filter
    """
    mask_size = mask_size = mask_size if mask_size else random.uniform(2000, 4000)
    mask_start = mask_start = mask_start if mask_start else random.uniform(500, 2500)
    freq_range = [mask_start, mask_start + mask_size]
    b, a = signal.butter(10, freq_range, "bandstop", fs=SAMPLING_FREQ)
    return signal.lfilter(b, a, input_arr)


def mask_high_freq(input_arr, mask_freq=None):
    """
    Randomly removes high frequency signal
    from input array using low pass filter
    """
    mask_freq = mask_freq if mask_freq else random.uniform(1500, 4000)
    b, a = signal.butter(10, mask_freq, "low", fs=SAMPLING_FREQ)
    return signal.lfilter(b, a, input_arr)


def mask_low_freq(input_arr, mask_freq=None):
    """
    Randomly removes high frequency signal
    from input array using low pass filter
    """
    mask_freq = mask_freq if mask_freq else random.uniform(500, 1500)
    b, a = signal.butter(10, mask_freq, "high", fs=SAMPLING_FREQ)
    return signal.lfilter(b, a, input_arr)
