import os
import pyFAI
import fabio
import shutil
import numpy as np
from xpdtools.cli.process_tiff import main as integrate
from typing import Iterable, List, Union, Tuple

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


# functions for integrations
def check_files(files: Union[str, Iterable[str]], ftype: str) -> List[str]:
    """
    check the type of files and make a list if it is a single string.
    :param files: single or a list of file names.
    :param ftype: the extension of the files that they should be.
    :return: a list of files.
    """
    # check type
    if isinstance(files, str):
        files = [files]
    # check existence and format
    for f in files:
        assert os.path.isfile(f), "file {} not exists".format(f)
        ext = "." + ftype
        assert os.path.splitext(f)[1] == ext, "{} does not have correct extension {}".format(f, ext)
    return files


def check_dir(dir_to_check: str) -> None:
    """
    check if directory exists. if not, create one.
    :param dir_to_check: path to the directory.
    :return: None.
    """
    # check existence, if not, create
    if os.path.isdir(dir_to_check):
        pass
    else:
        os.mkdir(dir_to_check)
        print("{} is created".format(dir_to_check))
    return


def check_kwargs(kwargs: dict, options: List[str]) -> None:
    """
    check if the keyword arguments are known.
    :param kwargs: keyword arguments.
    :param options: the known keyword.
    :return: None.
    """
    for key in kwargs.keys():
        if key in options:
            pass
        else:
            raise Exception("{} is an invalid keyword.".format(key))
    return


# functions for integration
def xpdtools_int(poni_file: str, tiff_file: str, chi_dir: str = None, **kwargs) -> Tuple[str, str]:
    """
    integrate using xpdtools and save result in chi file.
    Parameters
    ----------
    poni_file
        The path to poni file.
    tiff_file
        The path to tiff file.
    chi_dir
        Move the chi files to chi_dir if chi_dir is not None. Default None.
    kwargs
        The kwargs arguments to pass to integrate. They are:
        bg_file: str or None, optional
        Background image, if None no background subtraction is performed,
        defaults to None.
        mask_file: str or None, optional
            Mask file to include in the data processing, if None don't use one,
            defaults to None.
        polarization: float, optional
            The polzarization factor to use, defaults to .99, if None do not
            perform polarization correction
        edge: int, optional
            The number of pixels from the edge to mask with an edge mask,
            defaults to 20, if None no edge mask used
        lower_thresh: float, optional
            Threshold for lower threshold mask, all pixels with value less than
            this value (after background subtraction if applicable), defaults
            to 1. if None do not apply lower theshold mask
        upper_thresh: float, optional
            Threshold for upper threshold mask, all pixels with value greater
            than this value (after background subtraction if applicable),
            defaults to None if None do not apply upper theshold mask
        alpha: float, optional
            Number of standard deviations away from the ring mean to mask,
            defaults to 3. if None do not apply automated masking
        auto_type : {'median', 'mean'}, optional
            The type of automasking to use, median is faster, mean is more
            accurate. Defaults to 'median'.
        mask_settings: {'auto', 'first', none}, optional
            If auto mask every image, if first only mask first image, if None
            mask no images. Defaults to None
        flip_input_mask: bool, optional
            If True flip the input mask up down, this helps when using fit2d
            defaults to True.
        bg_scale : float, optional
            The scale for the image to image background subtraction, defaults
            to 1
    Returns
    -------
    path to chi file, path to mask file.
    """
    integrate(poni_file=poni_file, image_files=tiff_file, **kwargs)
    src_chi_file = os.path.splitext(tiff_file)[0] + ".chi"
    mask_file = os.path.splitext(tiff_file)[0] + "_mask.npy"

    # move them to chi_dir
    def move_to_dir():
        chi_file_name = os.path.basename(src_chi_file)
        dst_chi_file = os.path.join(chi_dir, chi_file_name)
        shutil.move(src_chi_file, dst_chi_file)
        return dst_chi_file

    if chi_dir:
        check_dir(chi_dir)
        moved_chi_file = move_to_dir()
    else:
        moved_chi_file = src_chi_file

    return moved_chi_file, mask_file


def pyfai_int(poni_file: str, tiff_file: str,
              xy_dir: str = None,
              bg_file: str = None,
              bg_scale: float = 1.,
              mask_file: str = None,
              invert_mask=True,
              **kwargs) -> str:
    """
    Integrate tiff image to Q, I array and save in xy file.
    Parameters
    ----------
    poni_file
        Path to the poni file.
    tiff_file
        Path to the tiff files.
    xy_dir
        Directory to output xy files. If None, xy file will be output in the directory where tiff file is. Default None.
    bg_file
        Path to the background tiff file. if None, subtract 0. Default None.
    bg_scale
        Scale for background subtraction. The data used in integration is tiff_data - bg_scale * bg_data. Default 1.0.
    mask_file
        Single or a list of npy mask files. Note that the auto-masking npy file output by xpdtools should be inverted
        before input to ai.integrate1d. If None: no masks. Default None.
    invert_mask
        Invert input mask or not. Only valid when mask_file is not None. Default True.
    kwargs
        kwargs for ai.integrate1d. The important parameters are:
        npt: number of points int the integration. Default 1480.
        polarization_factor: Apply polarization correction. If None: not applies. Default 0.99.
        correctSolidAngle: correct for solid angle. If None: not correct. Default None.
        method: integration method, a string or a registered method. Default "csr".
    Returns
    -------
    Path to xt file.
    """
    # get kwargs
    npt = kwargs.get("npt", 1480)
    unit = kwargs.get("unit", "q_A^-1")
    polarization = kwargs.get("polarization", 0.99)

    # create xy file name
    xy_file = os.path.splitext(tiff_file)[0] + ".xy"
    if xy_dir:
        check_dir(xy_dir)
        xy_file = os.path.join(xy_dir, os.path.basename(xy_file))
    else:
        pass

    # load background data
    bg_data = fabio.open(bg_file).data if bg_file else 0.0

    # integrate with pyFAI
    ai = pyFAI.load(poni_file)
    tiff_data = fabio.open(tiff_file).data

    if mask_file:
        mask_data = np.load(mask_file) == 1  # bool array
        if invert_mask:
            mask_data = np.invert(mask_data)
        else:
            pass
    else:
        mask_data = None

    ai.integrate1d(tiff_data - bg_scale * bg_data,
                   mask=mask_data,
                   filename=xy_file,
                   npt=npt, unit=unit, polarization_factor=polarization, correctSolidAngle=False)
    return xy_file


def invert_mask(mask: str) -> None:
    """
    invert the mask array in the mask file. true to false, false to true. pyfai and xpdtools have inverted mapping of
    masks so the mask generated by xpdtools is inverted before passing to the pyfai.
    :param mask: the npy file of mask array.
    :return: None.
    """
    # invert mask
    arr = np.load(mask)
    arr = np.invert(arr)
    # save mask file
    np.save(mask, arr)
    return
