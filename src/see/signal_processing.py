import numpy as np
from scipy import signal


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist

    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    result = signal.filtfilt(b, a, data)

    return result


def relative_wavelet_delay(w1: np.ndarray, w2: np.ndarray, dt: float = 1.0):
    """
    Get the relative delay between two wavelets with the same number of samples.

    Parameters
    ----------
    w1 : np.ndarray
        The first wavelet.
    w2 : np.ndarray
        The second wavelet.
    dt : float
        The time step
    """
    correlation = signal.correlate(w1, w2, mode="full")
    lags = signal.correlation_lags(len(w1), len(w2), mode="full")
    lag = lags[np.argmax(correlation)]

    return lag*dt
