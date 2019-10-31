import os
from typing import Iterable


def find_all_tiff(dir_path: str) -> Iterable[str]:
    """
    Yield the paths of all non-hidden tiff files in a directory.

    Parameters
    ----------
    dir_path
        path to the directory.

    Yields
    -------
    tiff_path:
        path to tiff files.

    """
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".tiff") and not filename.startswith("."):
                filepath = os.path.join(root, filename)
                yield filepath


def find_all(dir_path: str, ext: str) -> Iterable[str]:
    """
    Yield the paths of all non-hidden files with certain extension in a directory.

    Parameters
    ----------
    dir_path
        path to the directory.
    ext
        extension.

    Yields
    -------
    tiff_path:
        path to files.

    """
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(f".{ext}") and not filename.startswith("."):
                filepath = os.path.join(root, filename)
                yield filepath
