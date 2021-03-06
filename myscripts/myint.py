"""Integration tools."""
import os
import pyFAI
import fabio
import shutil
import numpy as np
import matplotlib.pyplot as plt
from xpdtools.cli.process_tiff import main as integrate
from typing import Iterable, List, Union, Tuple
from diffpy.pdfgetx import loaddata

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

__all__ = [
    "pipe_int",
    "xpdtools_int",
    "pyfai_int"
]


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
def pipe_int(poni_file: str, tiff_file: str, chi_dir: str = None, xy_dir: str = None, plot: bool = True,
             bg_file: str = None, bg_scale: float = 1.,
             xpdtools_config: dict = None, pyfai_config: dict = None):
    """
    Use the xpdtools to integrate the data and then use the mask it generated to apply to the data and integrate it
    with the pyfai.

    Parameters
    ----------
    poni_file
        The path to poni file.
    tiff_file
        The path to tiff file.
    chi_dir
        Move the chi files to chi_dir if chi_dir is not None. Default None.
    xy_dir
        Move the xy files to xy_dir if chi_dir is not None. Default None.
    plot
        Plot the qi data and mask image or not. Default True.
    bg_file
        Background image, if None no background subtraction is performed,
        defaults to None.
    bg_scale
        The scale for the image to image background subtraction, defaults
        to 1
    xpdtools_config
        The configuration for xpdtools_int. The allowed keys:
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
    pyfai_config
        The configuration for pyfai_int. The allowed keys:
        npt
            number of points int the integration. Default 1480.
        polarization_factor
            Apply polarization correction. If None: not applies. Default 0.99.
        correctSolidAngle
            correct for solid angle. If None: not correct. Default None.
        method
            integration method, a string or a registered method. Default "csr".

    Returns
    -------
    chi_file
        Path to the chi file. If chi_dir is specified, it will be in the chi_dir.
    xy_file
        Path to xy file. If xy_dir is specified, it will be in the xy_dir.
    """
    if xpdtools_config is None:
        xpdtools_config = {}
    if pyfai_config is None:
        pyfai_config = {}
    chi_file, mask_file_xpdtool = xpdtools_int(
        poni_file,
        tiff_file,
        chi_dir,
        plot=plot,
        bg_file=bg_file,
        bg_scale=bg_scale,
        **xpdtools_config
    )
    xy_file = pyfai_int(
        poni_file,
        tiff_file,
        xy_dir,
        mask_file=mask_file_xpdtool,
        plot=plot,
        bg_file=bg_file,
        bg_scale=bg_scale,
        **pyfai_config
    )
    return chi_file, xy_file


def xpdtools_int(poni_file: str, tiff_file: str, chi_dir: str = None, plot: bool = True, **kwargs) -> Tuple[str, str]:
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
    plot
        Plot the qi data and mask image or not. Default True.
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
    moved_chi_file
        Path to the chi file. If chi_dir is specified, it will be in the chi_dir.
    mask_file
        Path to the mask file. It is in the same directory as tiff file.
    """
    print(f"Integrate with xpdtools ...\nTiff file:\n{tiff_file}\nPoni file:\n{poni_file}")
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

    if plot:
        q, i = loaddata(moved_chi_file).T
        mask = np.load(mask_file) if os.path.exists(mask_file) else None
        tiff_data = fabio.open(tiff_file).data
        masked_data = np.ma.array(tiff_data, mask=np.invert(mask), fill_value=np.nan) if mask is not None else tiff_data
        plot_qi_and_mask(q, i, masked_data)
    else:
        pass

    return moved_chi_file, mask_file


def pyfai_int(poni_file: str, tiff_file: str,
              xy_dir: str = None,
              bg_file: str = None,
              bg_scale: float = 1.,
              mask_file: str = None,
              invert_mask: bool = True,
              plot: bool = True,
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
    plot
        Plot the results of integration or not. The mask will be plotted in a inverted situation. Default True.
    kwargs
        kwargs for ai.integrate1d. The important parameters are:
            npt
                number of points int the integration. Default 1480.
            polarization_factor
                Apply polarization correction. If None: not applies. Default 0.99.
            correctSolidAngle
                correct for solid angle. If None: not correct. Default None.
            method
                integration method, a string or a registered method. Default "csr".
    Returns
    -------
    xy_file
        Path to xy file. If xy_dir is specified, it will be in the xy_dir.
    """
    # get kwargs
    print(f"Integrate with PyFAI ...\nTiff file:\n{tiff_file}\nPoni file:\n{poni_file}")
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
        mask = np.load(mask_file) == 1  # bool array
        if invert_mask:
            mask = np.invert(mask)
        else:
            pass
    else:
        mask = None

    tiff_data = tiff_data - bg_scale * bg_data
    q, i = ai.integrate1d(tiff_data,
                          mask=mask,
                          filename=xy_file,
                          npt=npt, unit=unit, polarization_factor=polarization, correctSolidAngle=False)

    if plot:
        data_for_plot = np.ma.array(tiff_data, mask=mask, fill_value=np.nan) if mask is not None else tiff_data
        plot_qi_and_mask(q, i, data_for_plot)
    else:
        pass

    return xy_file


def _invert_mask(mask: str) -> None:
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


def plot_qi_and_mask(q: np.array, i: np.array, masked_data: np.ma.array = None):
    """
    Plot the qi curve and mask image on one figure.
    """
    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.plot(q, i, '-')
    plt.xlabel(r'Q ($\AA^{-1}$)')
    plt.ylabel(r'I (A. U.)')

    if masked_data is not None:
        plt.subplot(122)
        mean = masked_data.mean()
        std = masked_data.std()
        plt.imshow(masked_data, vmin=mean - 2 * std, vmax=mean + 2 * std)
    else:
        pass

    plt.show()
    return
