# coding=utf-8
import numpy as np
from diffpy.utils.parsers import loadData
import os
import re
from typing import Tuple, List, Union, Iterable
from matplotlib.pyplot import Figure, Axes
from matplotlib.lines import Line2D
from numpy import array
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat


def find(sdir: str, pattern: str) -> List[str]:
    """
    find files in directory that match the pattern.
    :param sdir: source directory.
    :param pattern: file name pattern.
    :return: a list of file paths whose file name match the pattern.
    """
    file_paths = []

    for item in os.listdir(sdir):
        path = os.path.join(sdir, item)
        if os.path.isfile(path) and re.match(pattern, item) is not None:
            file_paths.append(path)
        else:
            continue

    return file_paths


def recfind(sdir: str, pattern: str) -> List[str]:
    """
    find all files match the pattern recursively starting from the sdir.
    :param sdir: the source directory to find things.
    :param pattern: the pattern of the file name.
    :return: a list of file paths of the matched files.
    """
    file_paths = []

    for root, dir_names, file_names in os.walk(sdir):
        for file_name in file_names:
            if re.match(pattern, file_name):
                file_path = os.path.join(root, file_name)
                file_paths.append(file_path)
            else:
                continue

    return file_paths


def get_files(*files: str, sdir: str, exts: List[str]) -> List[str]:
    """
    get files from directory.
    :param files: file names to get.
    :param sdir: directory to search file.
    :param exts: allowable extension of files.
    :return: file paths.
    """
    def _find(tfile):
        path = None
        for root, dirs, sfiles in os.walk(sdir):
            for sfile in sfiles:
                sbase, sext = os.path.splitext(sfile)
                tbase, text = os.path.splitext(tfile)
                if sbase == tbase and (sext in exts or sext == text):
                    path = os.path.join(root, sfile)
                    break
                else:
                    continue
        if path is None:
            raise Exception(f"{tfile} is not found.")
        else:
            pass
        return path

    return [_find(f) for f in files]


# functions to check arguments
def check_files(files: Union[str, List[str]]) -> List[str]:
    """
    check if files is a list, if not, make it; check if file exists, if not, find it.
    :param files: files to check.
    :return: checked file list.
    """
    def find_file(target):
        res = None
        if os.path.isfile(target):
            res = target
        else:
            cdir = os.getcwd()
            for root, dirs, _files in os.walk(cdir):
                for _file_name in _files:
                    if _file_name == target:
                        res = os.path.join(root, _file_name)
                        break
                    else:
                        pass

        if res is None:
            raise Exception(f"File {target} is not found.")
        else:
            pass

        return res

    if isinstance(files, str):
        files = [files]
    else:
        pass

    checkedfiles = []
    for file_name in files:
        _file = find_file(file_name)
        checkedfiles.append(_file)

    return checkedfiles


def check_names(names: Union[None, str, List[str]], files: List[str]) -> List[str]:
    """
    check and modify the names to be a standard type for plotting.
    :param names: the argument names to check and modify.
    :param files: the file paths to plot.
    :return: modified names.
    """
    if names is None:
        checkednames = get_names(files)
    elif isinstance(names, str):
        checkednames = [names]
    elif len(names) == 0:
        checkednames = [""] * len(files)
    elif len(names) == len(files):
        checkednames = names
    else:
        print("Something wrong with names. Maybe length of names does not match length of files")
        checkednames = [""] * len(files)
    return checkednames


def check_colors(colors: Union[None, str, List[str]], flen: int) -> List[str]:
    """
    check and modify the colors to be a standard type for plotting.
    :param colors: the argument colors to check and modify.
    :param flen: the correct length of color.
    :return: modified colors.
    """
    if colors is None:
        checkedcolors = None
    elif isinstance(colors, str):
        checkedcolors = [colors]
    else:
        assert len(colors) == flen, "length of colors does not match length of files"
        checkedcolors = colors
    return checkedcolors


def check_lim(lim: Union[None, list, tuple], llen: int) -> Union[list, tuple]:
    """
    check and modify the colors to be a standard type for plotting.
    :param lim: a tuple to choose range of data.
    :param llen: the correct length of lim.
    :return: the same lim if there is no problems with lim.
    """
    if lim is None:
        checkedlim = lim
    else:
        assert isinstance(lim, (list, tuple)), "limit is not tuple."
        assert len(lim) == llen, "len of limit is not correct."
        checkedlim = lim
    return checkedlim


def check_kwargs(kwargs: dict, options: Union[list, tuple]) -> None:
    """
    check if the key word is known.
    :param kwargs: key word arguments.
    :param options: all the possible key words.
    :return:
    """
    for key in kwargs.keys():
        assert key in options, "Unknown keyword '{}'".format(key)
    return


# functions to get information
def get_names(files: Iterable) -> List[str]:
    """
    get base file name of a list of file paths.
    :param files: a list of file paths.
    :return: a list of base names.
    """
    names = []
    for f in files:
        base = os.path.basename(f)
        fname, _ = os.path.splitext(base)
        names.append(fname)
    return names


def get_rw(files: Iterable) -> List[str]:
    """
    get Rw from a list of res files.
    :param files: a list of file paths.
    :return: a list of Rw value in string form.
    """
    rws = []
    for f in files:
        rw = _get_rw(f)
        rws.append(rw)
    return rws


def _get_rw(file_path: str) -> str:
    """
    the function used in get_rw to get Rw value form a res file.
    :param file_path: path to res file.
    :return: Rw value in string form.
    """
    rw = ""
    with open(file_path, "r") as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            elif "Rw" in line:
                values = re.findall(r"\d+\.\d+", line)
                if len(values) == 1:
                    rw = values[0]
                    break
                elif len(values) > 1:
                    raise ValueError(f"more than one float in Rw line in {file_path}")
                else:
                    continue
            else:
                continue
    return rw


def attach_rw(names: List[str], rws: List[str], rwpos: str):
    """
    attach Rw value to the annotation.
    :param names: annotation of curves.
    :param rws: a list of Rw value in string.
    :param rwpos: how the Rw is attached to annotation.
    :return:
    """
    newnames = []
    for name, rw in zip(names, rws):
        rw_str = r"$R_w = {:.3f}$".format(float(rw))
        newname = name + rwpos + rw_str
        newnames.append(newname)
    return newnames


# functions in panel plotting
def split_horizontal(fig: Figure, rct: Tuple[float, float, float, float], frac: float) -> Tuple[Axes, Axes]:
    """
    create a axes in figure with two separate panels in horizontal.
    :param fig: figure to create axes.
    :param rct: lower left point x, lower left point y, width, height.
    :param frac: the fraction of left panel in width.
    :return: two axes.
    """
    x, y, w, h = rct
    x0, y0 = x, y
    w0, h0 = w * frac, h
    x1, y1 = x + w0, y
    w1, h1 = w - w0, h
    rct0 = (x0, y0, w0, h0)
    rct1 = (x1, y1, w1, h1)
    ax0 = fig.add_axes(rct0)
    ax1 = fig.add_axes(rct1)
    return ax0, ax1


def add_axes(fig: Figure, rct: Tuple[float, float, float, float], num: int) -> List[Axes]:
    """
    add axes to figure in a rectangular with several panels vertically.
    :param fig: figure to add axes.
    :param rct: lower left point x, lower left point y, width, height.
    :param num: number of panels.
    :return: a list of axes.
    """
    axs = []
    x0, y0, w, h = rct
    width = w
    height = h / num
    for n in range(num):
        xax = x0
        yax = y0 + h - (n + 1) * height
        ax = fig.add_axes([xax, yax, width, height])
        axs.append(ax)
    return axs


# functions to load data
def load_gr(files: List[str]) -> Tuple[List[array], List[array]]:
    """
    load r and g array from a list of data files and return a list of r array and a list of g array.
    :param files: path to data files.
    :return: r array list, g array list.
    """
    rs, gs = [], []
    for f in files:
        r, g = loadData(f).T
        rs.append(r)
        gs.append(g)
    return rs, gs


def load_xy(files: List[str]) -> Tuple[List[array], List[array]]:
    """
    load x and y array from a list of data files and return a list iof x array and a list of y array.
    :param files: a list of data files.
    :return: x array list, y array list.
    """
    xs, ys = [], []
    for f in files:
        x, y = loadData(f).T
        xs.append(x)
        ys.append(y)
    return xs, ys


def load_fgr(files: List[str]) -> Tuple[List[array], List[array], List[array], List[array]]:
    """
    load r array, g calculation array, g data array and difference between data and calculation array from a lit of
    data files and return four list of arrays.
    :param files: a list of data files.
    :return: r array list, gcalc array list, g array list, gdiff array list.
    """
    rs, gcalcs, gs, gdiffs = [], [], [], []
    for f in files:
        r, gcalc, g, _ = loadData(f).T
        gdiff = g - gcalc
        rs.append(r)
        gcalcs.append(gcalc)
        gs.append(g)
        gdiffs.append(gdiff)
    return rs, gcalcs, gs, gdiffs


def get_label(files):
    """
    decide what is x and y in plotting according to the extension of the first one in file list.
    :param files: a list of file paths.
    :return: the x label and y label.
    """
    files = iter(files)
    f = next(files)
    ext = os.path.splitext(f)[1]
    if ext == ".gr":
        xl, yl = r"r ($\AA$)", r"G ($\AA^{-2}$)"
    elif ext == ".fq":
        xl, yl = r"Q ($\AA^{-1}$)", r"F ($\AA^{-1}$)"
    elif ext == ".sq":
        xl, yl = r"Q ($\AA^{-1}$)", r"S"
    elif ext == ".iq":
        xl, yl = r"Q ($\AA^{-1}$)", r"I (A. U.)"
    else:
        xl, yl = r"x", r"y"
    return xl, yl


# function to prepare plotting data
def normalized(ys: List[array]) -> List[array]:
    """
    normalize all data arrays in a list so that their maximum of each one is always 1 but minimum can be lower than -1.
    :param ys: a list of data arrays.
    :return: a list of normalized data array.
    """
    return [y / abs(np.max(y)) for y in ys]


def normalize_fgr(gs: List[array], gcalcs: List[array], gdiffs: List[array]) -> None:
    """
    normalize g array, gcalc array, gdiff array in place.
    :param gs: a list of g array.
    :param gcalcs: a list of gcalc array.
    :param gdiffs: a list of gdiff array.
    :return: None.
    """
    for g, gcalc, gdiff in zip(gs, gcalcs, gdiffs):
        gmax = abs(np.max(g))
        g /= gmax
        gcalc /= gmax
        gdiff /= gmax
    return


def split_gr(rs: List[array], gs: List[array], rlim: Tuple[float, float, float]) \
        -> Tuple[List[array], List[array], List[array], List[array]]:
    """
    split a list of r and g to be two parts a low-r part and a high-r part.
    :param rs: a list of r array.
    :param gs: a list of g array.
    :param rlim: tuple of the start point, splitting point and the end point. splitting point is included in two parts.
    :return: low-r r array, low-r g array, high-r r array, high-r g array.
    """
    rs0, gs0, rs1, gs1 = list(), list(), list(), list()
    for r, g in zip(rs, gs):
        msk0 = np.logical_and(r >= rlim[0], r <= rlim[1])
        msk1 = np.logical_and(r >= rlim[1], r <= rlim[2])
        rs0.append(r[msk0])
        gs0.append(g[msk0])
        rs1.append(r[msk1])
        gs1.append(g[msk1])
    return rs0, gs0, rs1, gs1


def slice_data(xs:  List[array], ys: List[array], xlim: Tuple[float, float]) -> Tuple[List[array], List[array]]:
    """
    slice a list of xy data in a window defined by xlim.
    :param xs: a list of x array.
    :param ys: a list of y array.
    :param xlim: starting point (included) and end point (included).
    :return:
        sliced xs, sliced ys
    """
    xxs, yys = list(), list()
    for x, y in zip(xs, ys):
        msk = np.logical_and(x >= xlim[0], x <= xlim[1])
        xxs.append(x[msk])
        yys.append(y[msk])
    return xxs, yys


def shift_data(ys: List[array], spacing: float = 0.) -> None:
    """
    shift the y data so that the maximum point of the lower data is at same level as the minimum point of the higher
    data or with a spacing. the operation is in place.
    :param ys: a list of array that need to be shifted.
    :param spacing: the spacing of two adjacent data. Default 0.
    :return: None
    """
    for n in range(1, len(ys)):
        premin = np.min(ys[n-1])
        curmax = np.max(ys[n])
        shift = curmax - premin + spacing
        ys[n] -= shift
    return


def offset_fgr(gs: List[array], gcalcs: List[array], gdiffs: List[array], spacing: float = 0.) -> List[array]:
    """
    shift the difference curve so that the maximum of the curve is at the same level of the minimum of the minimums of
    calculation data and measured data or with a spacing. the operation is done to a list of data in place.
    :param gs: a list of g array.
    :param gcalcs: a list of gcalc array.
    :param gdiffs: a list of gdiff array.
    :param spacing: the spacing between top point of gdiff and the bottom point of previous minimum of g and gcalc.
    :return: a list of array that is at the level of zero of the shifted difference curve.
    """
    gzeros = []
    for gdiff, g, gcalc in zip(gdiffs, gs, gcalcs):
        gmin = min([g.min(), gcalc.min()])
        offset = gdiff.max() - gmin + spacing
        gdiff -= offset
        gzero = - offset * np.ones_like(gdiff)
        gzeros.append(gzero)
    return gzeros


def shift_fgr(gs: List[array], gcalcs: List[array], gdiffs: List[array], gzeros: List[array], spacing: float = 0.) \
        -> None:
    """
    shift a list the g, gcalc, gdiff, gzero so that the top point of the maximum of g and gcalc array is at the same
    level of the bottom point of the minimum the previous gdiff array or with a spacing. when the function is used,
    gdiff should be already offset to the level that beneath the g and gcalc. the operation is in place.
    :param gs: a list of g array.
    :param gcalcs: a list of gcalc array.
    :param gdiffs: a list of gdiff array.
    :param gzeros: a list of gzero array.
    :param spacing: the spacing between the top point of maximum of g and gcalc and the bottom point of gdiff.
    :return: None.
    """
    for n in range(1, len(gs)):
        premin = gdiffs[n - 1].min()
        curmax = max([gs[n].max(), gcalcs[n].max()])
        shift = abs(curmax - premin) + spacing
        gs[n] = gs[n] - shift
        gcalcs[n] -= shift
        gzeros[n] -= shift
        gdiffs[n] -= shift
    return


# functions to plot, annotate data
def add_circlines(axs: Union[Axes, List[Axes]], xs: List[array], ys: List[array]) -> List[Line2D]:
    """
    plot empty circular points on single axes or a list of axes to form a curve of y as function of x.
    :param axs: single axes or a list of axes.
    :param xs: a list of x array.
    :param ys: a list of y array.
    :return: a list of lines.
    """
    lines = list()
    if isinstance(axs, list):
        for ax, x, y in zip(axs, xs, ys):
            line, = ax.plot(x, y, "o", mfc="None")
            lines.append(line)
    else:
        for x, y in zip(xs, ys):
            line, = axs.plot(x, y, "o", mfc="None")
            lines.append(line)
    return lines


def add_solidlines(axs: Union[Axes, List[Axes]], xs: List[array], ys: List[array]) -> List[Line2D]:
    """
    plot solid lines on single axes or a list of axes to form a curve of y as a function x.
    :param axs: single axes or a list of axes.
    :param xs: a list of x array.
    :param ys: a list of y array.
    :return: a list of lines.
    """
    lines = list()
    if isinstance(axs, list):
        for ax, x, y in zip(axs, xs, ys):
            line, = ax.plot(x, y, ls="-")
            lines.append(line)
    else:
        for x, y in zip(xs, ys):
            line, = axs.plot(x, y, ls="-")
            lines.append(line)
    return lines


def add_fgrlines(axs, rs, gs, gcalcs, gdiffs, gzeros):
    ldatas, lcalcs, ldiffs, lzeros, fill_areas = list(), list(), list(), list(), list()
    if isinstance(axs, list):
        for ax, r, g, gcalc, gdiff, gzero in zip(axs, rs, gs, gcalcs, gdiffs, gzeros):
            ldata, = ax.plot(r, g, "o", mfc="None")
            lcalc, = ax.plot(r, gcalc, "-")
            lzero, = ax.plot(r, gzero, "--", color="grey")
            ldiff, = ax.plot(r, gdiff, "-")
            fill_area = ax.fill_between(r, gdiff, gzero, alpha=0.4)
            ldatas.append(ldata)
            lcalcs.append(lcalc)
            lzeros.append(lzero)
            ldiffs.append(ldiff)
            fill_areas.append(fill_area)
    else:
        ax = axs
        for r, g, gcalc, gdiff, gzero in zip(rs, gs, gcalcs, gdiffs, gzeros):
            ldata, = ax.plot(r, g, "o", mfc="None")
            lcalc, = ax.plot(r, gcalc, "-")
            lzero, = ax.plot(r, gzero, "--", color="grey")
            ldiff, = ax.plot(r, gdiff, "-")
            fill_area = ax.fill_between(r, gdiff, gzero, alpha=0.4)
            ldatas.append(ldata)
            lcalcs.append(lcalc)
            lzeros.append(lzero)
            ldiffs.append(ldiff)
            fill_areas.append(fill_area)
    return ldatas, lcalcs, ldiffs, lzeros, fill_areas


def annotate_plots(axs, names, poss, align=("left", "center")):
    annotations = []
    ha, va = align
    if isinstance(axs, list):
        for ax, name, pos in zip(axs, names, poss):
            annotation = ax.annotate(name, xy=pos, ha=ha, va=va)
            annotations.append(annotation)
    else:
        for name, pos in zip(names, poss):
            annotation = axs.annotate(name, xy=pos, ha=ha, va=va)
            annotations.append(annotation)
    return annotations


def add_legend(lines, names):
    for line, name in zip(lines, names):
        line.set_label(name)
    return


# functions to set color and set axes configuration
def calc_poss(xs, ys, fpos):
    poss = list()
    fx, fy = fpos
    # bound of positions is between the min of previous and max of current
    pre_ymin = np.max(ys[0]) - np.min(ys[0])
    for x, y in zip(xs, ys):
        xa = (1 - fx) * np.min(x) + fx * np.max(x)
        ya = (1 - fy) * np.max(y[x > xa]) + fy * np.min(pre_ymin)
        poss.append((xa, ya))
        # record the ymin for the next curve
        pre_ymin = np.min(y[x > xa])
    return poss


def calc_poss_fgr(xs, ys, ycalcs, ydiffs, fpos):
    poss = list()
    fx, fy = fpos
    # bound of positions is between the min of previous ydiff and max of current y and ycalc
    for n in range(len(xs)):
        xa = (1 - fx) * np.min(xs[n]) + fx * np.max(xs[n])
        curymax = max([np.max(ys[n][xs[n] > xa]), np.max(ycalcs[n][xs[n] > xa])])
        if n > 0:
            preydiffmin = np.min(ydiffs[n-1])
            ya = (1 - fy) * curymax + fy * preydiffmin
        else:
            _, ytop = plt.gca().get_ylim()
            ya = (1 - fy) * curymax + fy * ytop
        poss.append((xa, ya))
    return poss


def calc_eachposs(xs, ys, fpos):
    poss = list()
    fx, fy = fpos
    # bound of positions is between the min of previous and max of current
    for x, y in zip(xs, ys):
        xa = (1 - fx) * np.min(x) + fx * np.max(x)
        ya = (1 - fy) * np.max(y[x > xa]) + fy * np.max(y)
        poss.append((xa, ya))
        # record the ymin for the next curve
    return poss


def paint_color(arts, colors):
    for art, color in zip(arts, colors):
        art.set_color(color)
    pass


def get_comp(colors):
    # function for single item
    def _get_comp(color):
        if color[0] != "#":
            from matplotlib.colors import to_hex
            hexcolor = to_hex(color)[1:]
        else:
            hexcolor = color[1:]
        # convert the string into hex
        intcolor = int(hexcolor, 16)
        # invert the three bytes
        # as good as substracting each of RGB component by 255(FF)
        comp_color = 0xFFFFFF ^ intcolor
        # convert the color back to hex by prefixing a #
        comp_color = "#{:06X}".format(comp_color)
        # return the result
        return comp_color
    # run for every item
    return [_get_comp(c) for c in colors]


def config_ax(ax=None, xlim=None, ylim=None, xlabel=None, ylabel=None, minor=2):
    if ax:
        pass
    else:
        ax = plt.gca()
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if minor:
        ax.minorticks_on()
        from matplotlib.ticker import AutoMinorLocator
        ax.xaxis.set_minor_locator(AutoMinorLocator(minor))
        ax.yaxis.set_minor_locator(AutoMinorLocator(minor))
    pass


def config_panels(ax0, ax1, ax2, rlim, lpad):
    config_ax(ax0, xlim=(rlim[0], rlim[1]))
    config_ax(ax1, xlim=(rlim[1], rlim[2]))
    xlim = (rlim[0], rlim[2])
    ylim = ax0.get_ylim()
    config_ax(ax2, xlim=xlim, ylim=ylim)
    ax0.set_yticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlabel(r"r ($\AA$)", labelpad=lpad[0])
    ax2.set_ylabel(r"G ($\AA^{-2}$)", labelpad=lpad[1])
    pass


def config_axes(axs, rlim, xlabel, ylabel, minor=2):
    if rlim is not None:
        for ax in axs:
            ax.set_xlim(rlim[0], rlim[1])
    else:
        pass
    for ax in axs[:-1]:
        ax.set_xticklabels([])
    axs[-1].set_xlabel(xlabel)
    mid = int(len(axs) / 2)
    axs[mid].set_ylabel(ylabel)
    if minor is not None:
        for ax in axs:
            ax.minorticks_on()
            from matplotlib.ticker import AutoMinorLocator
            ax.xaxis.set_minor_locator(AutoMinorLocator(minor))
            ax.yaxis.set_minor_locator(AutoMinorLocator(minor))
    else:
        pass
    return


def higher_ylim(ax, frac):
    ymin, ymax = ax.get_ylim()
    ymax = - frac * ymin + (1 + frac) * ymax
    ax.set_ylim(ymin, ymax)
    return ymin, ymax


def label_panel(ax, text, fpos):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    fx, fy = fpos
    xa = (1 - fx) * xmin + fx * xmax
    ya = (1 - fy) * ymin + fy * ymax
    ax.annotate(text, xy=(xa, ya))
    pass


# function to save figures
def set_savefig(**settings):
    """
    Set kwargs for savefig and return a new function.

    Parameters
    ----------
    settings
        kwargs that can be passed to 'func'. Usually:
            caption
                caption to print in latex.
            fig
                figure to save. If None, get current figure.
            savedir
                directory to save the figure
            fmt
                format of the figure. It will also be the extension.
            print_tex
                choose to print latex or not.

    Returns
    -------
    _savefig
        a function to save figures.
    """
    def _savefig(filename, **kwargs):
        """
            Save current figure and print out the latex text of loading figure.

            Parameters
            ----------
            filename
                name of the file without extension.
            caption
                caption to print in latex.
            fig
                figure to save. If None, get current figure.
            savedir
                directory to save the figure
            fmt
                format of the figure. It will also be the extension.
            print_tex
                choose to print latex or not.

            Returns
            -------
                None.
            """
        return savefig(filename, **settings, **kwargs)
    return _savefig


def savefig(filename, caption="", fig=None, savedir=".", fmt="pdf", print_tex=True):
    """
    Save current figure and print out the latex text of loading figure.

    Parameters
    ----------
    filename
        name of the file without extension.
    caption
        caption to print in latex.
    fig
        figure to save. If None, get current figure.
    savedir
        directory to save the figure
    fmt
        format of the figure. It will also be the extension.
    print_tex
        choose to print latex or not.

    Returns
    -------
        None.
    """
    fig = fig if fig else plt.gcf()

    filename_with_ext = f"{filename}.{fmt}"
    filepath = os.path.join(savedir, filename_with_ext)

    fig.savefig(filepath, format=fmt)

    if print_tex:
        caption = "{" + caption + "}"
        label = "{" + "fig:" + filename + "}"
        file_in_tex = "{" + filename_with_ext + "}"
        tex = "\\begin{figure}[htb]\n" + \
              f"\\includegraphics[width=\\columnwidth]{file_in_tex}\n" + \
              f"\\caption{caption}\n" + \
              f"\\label{label}\n" + \
              "\\end{figure}"
        print(tex)

    return


# functions to print tables
def to_latex(df: pd.DataFrame, label="", caption="", **kwargs):
    """
    Convert pandas DataFrame to latex and print it out. kwargs will be passed to df.to_latex. Default, 'escape=False'.
    """
    kwargs["escape"] = kwargs["escape"] if "escape" in kwargs.keys() else False 
    tabular_str = df.to_latex(**kwargs)

    str_map = {r"\toprule": r"\hline\hline",
               r"\midrule": r"\hline",
               r"\bottomrule": r"\hline\hline"}
    for old, new in str_map.items():
        tabular_str = tabular_str.replace(old, new)

    label = "{" + "tab:" + label + "}"
    caption = "{" + caption + "}"
    table_str = "\\begin{table}[htb]\n" + \
                f"\\caption{caption}\n" + \
                f"\\label{label}\n" +\
                tabular_str +\
                "\\end{table}"

    print(table_str)
    return


# functions to deal with DataFrames
def join_result(csv_files: Iterable[str], chosen_column: str = 'val', column_names: Iterable[str] = None) -> pd.DataFrame:
    """
    Join multiple csv files into a single DataFrame with specific column names.

    Parameters
    ----------
    csv_files
        Multiple csv files.
    chosen_column
        Choose the column in each DataFrame to appear in the result. Default 'val'.
    column_names
        Column names for the resulting DataFrame.

    Returns
    -------
    df
        The result DataFrame.

    """
    dfs = (pd.read_csv(f, index_col=0)[chosen_column] for f in csv_files)
    df: pd.DataFrame = pd.concat(dfs, axis=1, ignore_index=True, sort=False)
    df = df.rename_axis(None)
    if isinstance(column_names, pd.Series):
        df.columns = column_names.to_list()
    elif column_names:
        df.columns = column_names
    else:
        pass
    return df


def join_result_with_std(csv_files: Iterable[str], column_names: Iterable[str] = None) -> pd.DataFrame:
    """
    Load multiple fitting results into a single DataFrame. Each cell contains 'value+/-std' (ufloat).

    Parameters
    ----------
    csv_files
        Multiple csv files.
    column_names
        Column names for the resulting DataFrame. If None, do not change column changes. Default None.

    Returns
    -------
    df
        The result DataFrame.

    """
    dfs = (pd.read_csv(f, index_col=0) for f in csv_files)

    def join_val_std(_df: pd.DataFrame):
        lst = [ufloat(val, std) for val, std in zip(_df['val'], _df['std'])]
        sr = pd.Series(lst, index=_df.index)
        return sr

    srs = [join_val_std(df) for df in dfs]
    res: pd.DataFrame = pd.concat(srs, axis=1, ignore_index=True, sort=False)
    res = res.rename_axis(None)
    if isinstance(column_names, pd.Series):
        res.columns = column_names.to_list()
    elif column_names:
        res.columns = column_names
    else:
        pass
    return res


def gradient_color(color1, color2, num):
    """Generate a list of long hex representations of gradient colors."""
    from colour import Color
    return [color.hex_l for color in Color(color1).range_to(Color(color2), num)]


def add_color_column(df: pd.DataFrame, src_df: pd.DataFrame, ref_col: str = 'name', src_col: str = 'label',
                     clr_col: str = 'c') -> None:
    """
    Add a column of assigned color to the DataFrame for plotting. The color is defined by mapping the value in the
    reference column in the DataFrame to the color column in the source DataFrame.

    Parameters
    ----------
    df
        The DataFrame to add color column.
    src_df
        The source DataFrame to find the color for each row.
    ref_col
        The name of the reference column in the DataFrame
    src_col
        The corresponding column in the source DataFrame.
    clr_col
        The name of the new column of colors.

    Returns
    -------
    None
        The operation is in place.

    """
    new_col = df[ref_col].apply(lambda cell_val: src_df.set_index(src_col).loc[cell_val, clr_col])
    df.insert(len(df.columns), clr_col, new_col)
    return
