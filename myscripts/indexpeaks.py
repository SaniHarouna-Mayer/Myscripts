import scipy.signal as signal
import numpy as np
from myscripts.helper import load_xy, slice_data
from typing import Tuple


__all__ = ["extract_peaks", "load_data", "save_data"]


def extract_peaks(data_file: str, output_file: str = None, limits: Tuple[float, float] = None, **kwargs)\
        -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Extract the peak positions and intensities in the data inside a data file. Save the results in a .txt file.

    Parameters
    ----------
    data_file
        The path to the data file. The data file contain multi-lines header and two columns.
    output_file
        The file name of the output file without extensions.
    limits
        The tuple of low limit of x and high limit of x.
    kwargs
        The keywords used in the scipy.signal.find_peaks function.

    Returns
    -------
    x_data
        The numpy array of the data in the first column in the data file.
    y_data
        The numpy array of the data in the second column in the data file.
    x_peaks
        The numpy array of the x positions of the peaks.
    y_peaks
        The numpy array of the y intensities of the peaks.

    """
    x_data, y_data = load_data(data_file, limits)
    ind_peaks, _ = signal.find_peaks(y_data, **kwargs)
    x_peaks, y_peaks = x_data[ind_peaks], y_data[ind_peaks]
    if output_file is not None:
        save_data(output_file, x_peaks, y_peaks, **kwargs)
    return x_data, y_data, x_peaks, y_peaks


def load_data(data_file: str, limits: tuple = None) -> Tuple[np.array, np.array]:
    """
    Load data from a data file, slice them and return them out.

    Parameters
    ----------
    data_file
        The path to the data file. The data file contain multi-lines header and two columns.
    limits
        The tuple of low limit of x and high limit of x.

    Returns
    -------
    x
        The array of the first column of the data.
    y
        The array of the second column of the data.

    """
    xs, ys = load_xy([data_file])
    if limits is not None:
        xs, ys = slice_data(xs, ys, limits)
    x, y = xs[0], ys[0]
    return x, y


def save_data(output_file: str, x_peaks: np.array, y_peaks: np.array, **kwargs) -> None:
    """
    Save the extracted peak positions and intensity as two columns of data in a .txt file with the header of peak
    extraction settings.

    Parameters
    ----------
    output_file
        The file name of the output file.
    x_peaks
        The numpy array of the peaks positions.
    y_peaks
        The numpy array of the peaks intensities.
    kwargs
        The keywords used in the scipy.signal.find_peaks function.

    Returns
    -------
    None

    """
    header = "\n".join([f"{key}: {value}" for key, value in kwargs.items()])
    header += "\nx_peaks y_peaks"
    np.savetxt(output_file, np.column_stack((x_peaks, y_peaks)), header=header)
    return
