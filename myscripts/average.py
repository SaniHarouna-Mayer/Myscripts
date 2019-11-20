import os
import shutil
import fabio
import fire
import numpy as np
import matplotlib.pyplot as plt
from myscripts.tools import find_all_tiff, find_all
from myscripts.helper import loadData
from typing import Tuple


__all__ = ["main"]


def average_tiff(dir_path: str, outfile: str, tiff_names: Tuple[str] = None):
    """
    Average the tiff images in a directory and save it as a tiff file in that directory.

    Parameters
    ----------
    dir_path
        The path to the directory to find tiffs.
    outfile
        The output file name.
    tiff_names
        (Optional) A list of tiff file names to average. If None, every tiff file in the folder will be used.

    Returns
    -------
    None

    """
    os.chdir(dir_path)

    if tiff_names:
        tiff_paths = tiff_names
    else:
        tiff_paths = list(find_all_tiff(dir_path))

    outfile_path = outfile
    if os.path.exists(outfile_path):
        raise ValueError("The outfile has the same name as one of the data!")
    else:
        shutil.copy(tiff_paths[0], outfile_path)

    num_data = 0
    sum_data = 0
    total_intensities = []
    for tiff_path in tiff_paths:
        data: np.array = fabio.open(tiff_path).data
        sum_data: np.array = sum_data + data
        num_data: int = num_data + 1
        total_intensity = np.sum(data)
        total_intensities.append(total_intensity)

    tiff_obj = fabio.open(outfile_path)
    tiff_obj.data = sum_data / num_data
    tiff_obj.save(outfile_path)

    plt.figure(figsize=(8, 4))
    plt.axes([.1, .2, .3, .6])
    plt.bar(range(len(total_intensities)), sorted(total_intensities))
    plt.axes([.6, .2, .3, .6])
    plt.imshow(sum_data)
    plt.show()

    return


def average_files(dir_path: str, outfile: str, ext: str = None, file_names: Tuple[str] = None):
    """
    Average the txt files of two columns of data in a directory and save it as a txt file in that directory.

    Parameters
    ----------
    dir_path
        The path to the directory to find tiffs.
    outfile
        The output file name.
    ext
        The extension of files.
    file_names
        (Optional) A list of file names to average. If None, every txt files in the folder will be used.

    Returns
    -------
    None

    """
    os.chdir(dir_path)

    ext = ext if ext else ".chi"
    file_names = file_names if file_names else find_all(dir_path, ext)

    xavg = 0
    yavg = 0
    counter = 0
    for file_name in file_names:
        x, y = loadData(file_name).T
        plt.plot(x, y)
        xavg += x
        yavg += y
        counter += 1

    xavg /= counter
    yavg /= counter
    np.savetxt(outfile, np.column_stack([xavg, yavg]))

    plt.show()
    return


def main(outfile: str, *file_names: str, file_type: str = None, dir_path: str = None):
    """
    Average tiff image or integration results.
    Parameters
    ----------
    outfile
        The output file name.
    file_type
        (Optional) File type to average. Options: 'tiff', 'xy', 'chi', 'iq', 'sq', 'fq', 'gr'. If None, use the same
        file type of outfile according to the extension.
    dir_path
        (Optional) The path to the directory to find tiffs. If None, use './'
    file_names
        (Optional) File names to average.

    Returns
    -------
    None

    """
    if file_type:
        pass
    else:
        ext = os.path.splitext(outfile)[1]
        if len(ext) > 0:
            file_type = ext[1:]  # no '.' ahead
        else:
            raise ValueError("Cannot decide file type according to the outfile without extension.")

    dir_path = dir_path if dir_path else "./"

    if file_type == "tiff":
        average_tiff(dir_path, outfile, file_names)
    elif file_type in ("xy", "chi", "iq", "sq", "fq", "gr"):
        average_files(dir_path, outfile, file_type, file_names)
    else:
        raise ValueError(f"Unknown file type: {file_type}")
    return


if __name__ == "__main__":
    fire.Fire(main)
