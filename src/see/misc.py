import os
import numpy as np


def find_nearest(array: object, value: float) -> tuple:
    """
    Find the nearest value in an array.

    Parameters
    ----------
    array : object
        The array to search in.
    value : float
        The value to search for.

    Returns
    -------
    tuple
        The nearest value and its index.
    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return array[idx], int(idx)


def get_filenames_from_directory(directory: str, extension: str = None, remove_extension=True) -> list:
    """
    Get the filenames from a directory.

    Parameters
    ----------
    directory : str
        The directory to get the filenames from.
    extension : str, optional
        The extension of the files to get. The default is None.

    Returns
    -------
    list
        The list of filenames.
    """
    filenames = os.listdir(directory)

    if extension is not None:
        filenames = [filename for filename in filenames if filename.endswith(extension)]

    if remove_extension:
        filenames = [filename.split('.')[0] for filename in filenames]

    return filenames
