import os
import numpy as np
import h5py as h5
from scipy import signal
from typing import Dict
from .constants import PZT_ACTIVATION_DELAY
from .signal_processing import butter_lowpass_filter
from .misc import get_filenames_from_directory


class Trace:
    """
    A trace is a set of data points (time and amplitude).
    """

    def __init__(self, time: np.ndarray, amplitude: np.ndarray):
        self.time = time
        self.amplitude = amplitude
        self.dt = time[1] - time[0]


def read_keysight_h5(file_path: str, channel_number="1", is_seismic=False) -> Trace:
    """
    Read a Keysight H5 file.
    """
    f = h5.File(file_path, "r")
    channel = f[f'Waveforms/Channel {channel_number}']

    x0 = channel.attrs.get('XOrg')
    y0 = channel.attrs.get('YOrg')
    dx = channel.attrs.get('XInc')
    dy = channel.attrs.get('YInc')
    num_points = channel.attrs.get('NumPoints')

    data = f[f'Waveforms/Channel {channel_number}/Channel {channel_number}Data']

    amplitude = dy*data + y0
    time = dx*np.arange(num_points) + x0
    # Remove PZT activation delay.
    time = time - PZT_ACTIVATION_DELAY

    # Remove origin offset to center signal in 0.
    amplitude = amplitude - amplitude[0]

    if is_seismic:
        # Converts voltage to displacement in meters (m) using the convertion factor of 50 nm/V
        # given by the laser datasheet.
        amplitude = amplitude*50*1e-9
    else:
        # Electric potential tends to be heavily polluted by noise.
        # Remove high frequency (over 600 kHz) noise.
        amplitude = butter_lowpass_filter(data=amplitude, fs=1/dx, cutoff=600e3, order=3)

        # Remove the linear trend.
        amplitude = signal.detrend(amplitude)

    f.close()

    return Trace(time.astype(np.float32), amplitude.astype(np.float32))


def read_see_data(file_path: str, is_seismic=False) -> Trace:
    """
    Read the SEE data from a file.

    Parameters
    ----------
    file_path : str
        The file to read the SEE data from.

    Returns
    -------
    Trace
        The trace data.
    """
    # We always record the seismic signal in channel 8 and the electric potential in channel 1.
    channel_number = "8" if is_seismic else "1"

    trace = read_keysight_h5(file_path=file_path, channel_number=channel_number, is_seismic=is_seismic)

    return trace


def read_see_data_directory(directory: str) -> Dict[str, Trace]:
    """
    Read the SEE data from a directory.

    Parameters
    ----------
    directory : str
        The directory to read the SEE data from.

    Returns
    -------
    Dict[str, Trace]
        The SEE data.
    """
    filenames = get_filenames_from_directory(directory, extension='.h5', remove_extension=False)

    data = {}

    for filename in filenames:
        is_seismic = filename.startswith("u") or filename.startswith("source")

        trace = read_see_data(os.path.join(directory, filename), is_seismic)

        data[filename.split('.')[0]] = trace

    return data
