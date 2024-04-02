import numpy as np
from scipy import signal


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    result = signal.filtfilt(b, a, data)

    return result
