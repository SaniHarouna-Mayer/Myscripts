import os
import re
from myscripts.helper import recfind
import yaml
from typing import List


def make_dict(data_path: str, poni_mapping: dict = None, bg_mapping: dict = None) -> dict:
    """
    Make yaml for integration, including four columns: samples, tiffs, ponis, bgs. The yaml store a dict with folder
    names as keys and the tiff images list in it, the poni file mapping to it and the background tiff mapping to it as
    values.
    :param data_path: a string that is the path to the data directory
    :param poni_mapping: a dictionary mapping a strings to poni file paths.
           If string is in directory name, all tiff files in the directory will be associated with the poni file path.
    :param bg_mapping: a dictionary mapping a string to background file path.
           If string is in directory name, all tiff files in the directory will be associated with the bg file path.
    :return: pandas DataFrame object.
    """
    sample_dirs = sorted([os.path.join(data_path, d) for d in os.listdir(data_path) if d[0] != "."])

    dct = {}
    for sample_dir in sample_dirs:
        sample = os.path.basename(sample_dir)
        dct[sample] = {}

        indir_tiffs = recfind(sample_dir, r"^(?!\.).*\.tiff")
        dct[sample]["tiff_files"] = sorted(indir_tiffs)

        indir_yamls = recfind(sample_dir, r"^(?!\.).*\.yaml")
        dct[sample]["yaml_files"] = sorted(indir_yamls)

        poni = _dirname_to_poni(sample_dir, poni_mapping)
        dct[sample]["poni_file"] = poni

        bg_files = _dirname_to_bg_files(data_path, sample_dir, bg_mapping)
        dct[sample]["bg_files"] = sorted(bg_files) if bg_files else None

    return dct


def _dirname_to_poni(dirname: str, mapping: dict = None) -> str:
    """
    return the path to poni file according to the name of the directory and a dictionary of mapping.
    :param dirname: name of the directory.
    :param mapping: how the name of dictionary is mapped to path of poin file.
    :return:
    """
    if mapping:
        for pattern in mapping.keys():
            if re.match(pattern, dirname):
                poni = mapping[pattern]
                break
        else:
            print(f"None of the poni patterns can be mapped to the directory: {dirname}")
            poni = None
    else:
        poni = None
    return poni


def _dirname_to_bg_files(dirs_root: str, dirname: str, mapping: dict = None) -> List[str]:
    """
    Return the path to background files by searching the background directory, which is determined by mapping sample
    directory name.
    Parameters
    ----------
    dirs_root
        Root of the sample directory and background directory.
    dirname
        Name of the directory of sample. It can be a python re pattern.
    mapping
        A dictionary that maps name of sample directory to the name of the background directory.
    Returns
    -------

    """
    if mapping:
        for pattern in mapping.keys():
            if re.match(pattern, dirname):
                bg_dirname = mapping[pattern]
                bg_dirpath = os.path.join(dirs_root, bg_dirname)
                bg_files = recfind(bg_dirpath, r"^(?!\.).*\.tiff")
                break
        else:
            print(f"None of the bg patterns can be mapped to the directory: {dirname}")
            bg_files = None
    else:
        bg_files = None
    return bg_files


# functions for load yaml
def load(yaml_file: str) -> dict:
    """
    load yaml file to python object.
    """
    with open(yaml_file, "r") as f:
        dct = yaml.safe_load(f)
    return dct


def dump(dct: dict, yaml_file: str) -> None:
    """
    dump python object into yaml file.
    """
    with open(yaml_file, "w") as f:
        yaml.safe_dump(dct, f)
    return
