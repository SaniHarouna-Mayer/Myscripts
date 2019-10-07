import matplotlib.pyplot as plt
from myscripts.helper import *


def compare_data(files=None, rlim=None, names=None, colors=None, normal=False, diff=False):
    """
    plot y vs. x as lines for comparison.
    :param files: a list of strings that are paths of the data file.
    :param rlim: a tuple of two floats that are endpoints of x-range.
    :param names: a list of strings that are annotations for lines.
                  If names is empty list, no names are added, elif names is None, names are file names.
    :param colors: a list of strings that are hex value of rgb color for lines.
    :param normal: a bool variable to turn on and off the normalization. Default is False.
    :param diff: a bool variable to turn on the difference curve. Default is False.
    :return: a matpotlib figure object.
    """
    files = check_files(files)
    names = check_names(names, files)
    colors = check_colors(colors, len(files))
    rlim = check_lim(rlim, 2)

    # load data
    xs, ys = load_xy(files)

    # slice data
    if rlim:
        xs, ys = slice_data(xs, ys, rlim)
    else:
        pass
    # normal
    if normal:
        ys = normalized(ys)
    else:
        pass
    # initiate figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot
    lines = add_solidlines(ax, xs, ys)
    # difference
    if diff:
        assert len(files) == 2, f"diff is only valid for two files but input is {len(files)} files"
        xdiffs = [xs[0]]
        ydiffs = [ys[1] - ys[0]]
        offset_fgr(ys, ys, ydiffs)
        _ = add_solidlines(ax, xdiffs, ydiffs)
    # colors
    if colors is not None:
        paint_color(lines, colors)
    else:
        pass
    # legends
    add_legend(lines, names)
    ax.legend()
    # check file type and get labelss
    xl, yl = get_label(files)
    config_ax(ax, rlim, None, xl, yl)
    return fig


def compare_panel(files, rlim, names=None, colors=None, normal=False, **kwargs):
    """
    plot y vs. x as lines in two panels of low-x and high-x regions for comparison.
    :param files: a list of strings that are paths to the data files.
    :param rlim: a tuple of three floats that are the left, middlem, right points for the r-range
    :param names: a list of strings that are annotations for lines.
                  If names is empty list, no names are added, elif names is None, names are file names.
    :param colors: a list of strings that are hex values starting with "#" for colors.
    :param normal: a bool value to turn on and off the normalization for data. Default is False.
    :param kwargs: arguments of positions for annotation, label and configuration of limits. They are
        frac: a float that is the fraction between left panel width and whole width.
              Default is .1
        lpos: a tuple of two floats as fractional position of panel label in data frame.
              Default is (.05, .95)
        yinc: a float as fractional increase of ymax to leave margin for figure label.
              Default is .1
        lpad: a tuple of two ints that are labelpad for xlabel and ylabel.
              Default is (25, 0)
    :return: a matpotlib figure object.
    """
    # check arguments
    files = check_files(files)
    rlim = check_lim(rlim, 3)
    names = check_names(names, files)
    colors = check_colors(colors, len(files))
    # set kwargs
    default_frac = (rlim[1] - rlim[0]) / (rlim[2] - rlim[0])
    frac = kwargs.get("frac", default_frac)
    lpos = kwargs.get("lpos", (.05, .95))
    yinc = kwargs.get("yinc", .1)
    lpad = kwargs.get("lpad", (25, 0))
    # load data
    rs, gs = load_gr(files)
    # normal
    if normal:
        gs = normalized(gs)
    else:
        pass
    # split data
    rs0, gs0, rs1, gs1 = split_gr(rs, gs, rlim)
    # initiate figure
    fig = plt.figure()
    rct = (.1, .1, .8, .8)
    ax0, ax1 = split_horizontal(fig, rct, frac)
    ax2 = fig.add_axes(rct, frameon=False)
    # plot
    lines0 = add_solidlines(ax0, rs0, gs0)
    lines1 = add_solidlines(ax1, rs1, gs1)
    # colors
    if colors is not None:
        paint_color(lines0, colors)
        paint_color(lines1, colors)
    else:
        pass
    # set higher ymax
    ymin0, ymax0 = higher_ylim(ax0, yinc)
    ymin1, ymax1 = higher_ylim(ax1, yinc)
    scale = (ymax0 - ymin0) / (ymax1 - ymin1)
    label0 = r"(a)"
    label1 = r"(b) scale $\times$ {:.2f}".format(scale)
    label_panel(ax0, label0, lpos)
    label_panel(ax1, label1, lpos)
    # configure ax
    config_panels(ax0, ax1, ax2, rlim, lpad)
    # annotation
    add_legend(lines1, names)
    ax1.legend()
    return fig


def plot_gr(files, rlim=None, names=None, colors=None, normal=False, **kwargs):
    """
    waterfall plot g vs. r in empty circles.
    :param files: a list of strings that are paths to the data files.
    :param rlim: a tuple of two floats that are endpoints of data.
    :param names: a list of strings that are annotations for lines.
                  If names is empty list, no names are added, elif names is None, names are file names.
    :param colors: a list of strings that are hex values starting with "#" for colors.
    :param normal: a bool value to turn on and off the normalization for data. Default is False.
    :param kwargs: kwargs for plot settings. They are
           style: a string to choose plotting curve style. It can be "o" for empty circle and "-" for solid lines.
                  Default is "-".
           apos: a tuple of two floats as fractional posistion of annotation in data frame. Default is (.8, .5).
    :return: a matpotlib figure object.
    """
    files = check_files(files)
    rlim = check_lim(rlim, 2)
    names = check_names(names, files)
    colors = check_colors(colors, len(files))
    style = kwargs.get("style", "-")
    apos = kwargs.get("apos", (.8, .5))

    rs, gs = load_gr(files)

    if rlim:
        rs, gs = slice_data(rs, gs, rlim)
    else:
        pass

    if normal:
        gs = normalized(gs)
    else:
        pass

    shift_data(gs)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if style == "o":
        lines = add_circlines(ax, rs, gs)
    elif style == "-":
        lines = add_solidlines(ax, rs, gs)
    else:
        raise ValueError("Unknown style: {}".format(style))

    poss = calc_poss(rs, gs, apos)
    anns = annotate_plots(ax, names, poss)

    if colors is not None:
        paint_color(lines, colors)
        paint_color(anns, colors)
    else:
        pass

    config_ax(ax, rlim, None, r'r ($\AA$)', r'G ($\AA^{-2}$)')

    return fig


def plot_panel(files, rlim, names=None, colors=None, normal=False, **kwargs):
    """
    water fall plot g vs. r in two panels of low-r and high-r regions.
    :param files: a list of strings that are paths to the data files.
    :param rlim: a tuple of three floats that are the left, middlem, right points for the r-range
    :param names: a list of strings that are annotations for lines.
                  If names is empty list, no names are added, elif names is None, names are file names.
    :param colors: a list of strings that are hex values starting with "#" for colors.
    :param normal: a bool value to turn on and off the normalization for data. Default is False.
    :param kwargs: arguments of positions for annotation, label and configuration of limits. They are
        style: a string for plotting curve style. It can be "o" for empty circle and "-" for solid lines.
               Default is "-".
        frac: a float that is the fraction between left panel width and whole width.
              Default is the ratio between the width of low-r region and high-r reigon.
        apos: a tuple of two floats as fractional position of annotation in data frame of the line.
              Default is (.80, .40)
        lpos: a tuple of two floats as fractional position of panel label in data frame.
              Default is (.05, .95)
        yinc: a float as fractional increase of ymax to leave margin for figure label.
              Default is .1
        lpad: a tuple of two ints that are labelpad for xlabel and ylabel.
              Default is (25, 0)
    :return: a matpotlib figure object.
    """
    # check arguments
    files = check_files(files)
    rlim = check_lim(rlim, 3)
    names = check_names(names, files)
    colors = check_colors(colors, len(files))
    # set kwargs
    style = kwargs.get("style", "-")
    default_frac = (rlim[1] - rlim[0]) / (rlim[2] - rlim[0])
    frac = kwargs.get("frac", default_frac)
    apos = kwargs.get("apos", (.80, .40))
    lpos = kwargs.get("lpos", (.05, .95))
    yinc = kwargs.get("yinc", .1)
    lpad = kwargs.get("lpad", (25, 0))
    # load data
    rs, gs = load_gr(files)
    # normal
    if normal:
        gs = normalized(gs)
    else:
        pass
    # split data
    rs0, gs0, rs1, gs1 = split_gr(rs, gs, rlim)
    # shift data
    shift_data(gs0)
    shift_data(gs1)
    # initiate figure
    fig = plt.figure()
    rct = (.1, .1, .8, .8)
    ax0, ax1 = split_horizontal(fig, rct, frac)
    ax2 = fig.add_axes(rct, frameon=False)
    # plot
    if style == "o":
        lines0 = add_circlines(ax0, rs0, gs0)
        lines1 = add_circlines(ax1, rs1, gs1)
    elif style == "-":
        lines0 = add_solidlines(ax0, rs0, gs0)
        lines1 = add_solidlines(ax1, rs1, gs1)
    else:
        raise ValueError("Unknown style: {}".format(style))
    # add annotations
    poss = calc_poss(rs1, gs1, apos)
    anns = annotate_plots(ax1, names, poss)
    # add names and colors
    if colors is not None:
        paint_color(anns, colors)
        paint_color(lines0, colors)
        paint_color(lines1, colors)
    else:
        pass
    # set higher ymax
    ymin0, ymax0 = higher_ylim(ax0, yinc)
    ymin1, ymax1 = higher_ylim(ax1, yinc)
    scale = (ymax0 - ymin0) / (ymax1 - ymin1)
    label0 = r"(a)"
    label1 = r"(b) scale $\times$ {:.2f}".format(scale)
    label_panel(ax0, label0, lpos)
    label_panel(ax1, label1, lpos)
    # configure ax
    config_panels(ax0, ax1, ax2, rlim, lpad)
    return fig


def plot_fgr(files, rlim=None, names=None, colors=None, normal=False, **kwargs):
    """
    waterfall plot of fitted g vs. r with difference curve and zeroline below.
    :param files: a list of strings that are paths to the fgr files.
    :param rlim: a tuple of two floats that are endpoints of r-range.
    :param names: a list of strings that are annotations for lines.
                  If names is empty list, no names are added, elif names is None, names are file names.
    :param colors: a list of strings starting with "#" that are hex value of rgb colors for data and difference.
                   fitted lines will be in complementary color of data line.
    :param normal: a bool variable to turn on and off the normalization for data. It makes data line to peak at 1.0
                   while fitted and differene lines are scaled by same ratio.
    :param kwargs: kwargs for parameters in plotting to make figure beautiful. They are
        res_scale: (float) scale factor for residuals. If None, no scaling. Default None
        spacing: (float) the spacing of data-difference and difference-data in data value.
                 Default is 0.
        apos: (Tuple[float, float])position of annotation in data frame.
              Default is (.85. 50)
        auto_rw: (bool) choose if the rw is automatically read and added to the annotation. Default is False.
        rwpos: (string) choose how rw is attached to the names. Default is "\n".
    :return: a matpotlib figure object.
    """
    # check args
    files = check_files(files)
    rlim = check_lim(rlim, 2)
    names = check_names(names, files)
    colors = check_colors(colors, len(files))

    # check kwargs
    options = ["spacing", "apos", "auto_rw", "rwpos", "rws", "res_scale"]
    check_kwargs(kwargs, options)
    spacing = kwargs.get("spacing", 0.)
    apos = kwargs.get("apos", (.85, .50))
    auto_rw = kwargs.get("auto_rw", False)
    rwpos = kwargs.get("rwpos", "\n")
    rws = kwargs.get("rws", None)
    res_scale = kwargs.get("res_scale", None)

    # get rw
    if auto_rw:
        rws_from_file = get_rw(files)
        names = attach_rw(names, rws_from_file, rwpos)
    elif rws is not None:
        names = attach_rw(names, rws, rwpos)
    else:
        pass

    # load data
    rs, gcalcs, gs, gdiffs = load_fgr(files)
    if res_scale:
        gdiffs = [gdiff * res_scale for gdiff in gdiffs]

    # slice data
    if rlim:
        for n, (r, gcalc, g, gdiff) in enumerate(zip(rs, gcalcs, gs, gdiffs)):
            mask = np.logical_and(r >= rlim[0], r <= rlim[1])
            rs[n] = r[mask]
            gs[n] = g[mask]
            gcalcs[n] = gcalc[mask]
            gdiffs[n] = gdiff[mask]

    # normal
    if normal:
        normalize_fgr(gs, gcalcs, gdiffs)
    else:
        pass
    # offset gdiffs
    gzeros = offset_fgr(gs, gcalcs, gdiffs, spacing)

    # shift and record zeros
    shift_fgr(gs, gcalcs, gdiffs, gzeros, spacing)

    # initiate figure
    fig = plt.gcf()
    ax = plt.gca()

    # initiate lines
    ldatas, lcalcs, ldiffs, lzeros = add_fgrlines(ax, rs, gs, gcalcs, gdiffs, gzeros)

    # add annotations
    if apos is None:
        pass
    else:
        poss = calc_poss(rs, gs, apos)
        anns = annotate_plots(ax, names, poss)
        if colors is None:
            pass
        else:
            paint_color(anns, colors)

    # set names and colors
    if colors is None:
        pass
    else:
        paint_color(ldatas, colors)
        comp_colors = get_comp(colors)
        paint_color(lcalcs, comp_colors)
        paint_color(ldiffs, colors)

    # configure axes
    config_ax(ax, rlim, None, r"r ($\AA$)", r"G ($\AA^{-2}$)")
    return fig


def plot_axfgr(files, rlim=None, names=None, colors=None, normal=False, **kwargs):
    """
        waterfall plot of fitted g vs. r with difference curve and zeroline below.
        :param files: a list of strings that are paths to the fgr files.
        :param rlim: a tuple of two floats that are endpoints of r-range.
        :param names: a list of strings that are annotations for lines.
                      If names is empty list, no names are added, elif names is None, names are file names.
        :param colors: a list of strings starting with "#" that are hex value of rgb colors for data and difference.
                       fitted lines will be in complementary color of data line.
        :param normal: a bool variable to turn on and off the normalization for data. It makes data line to peak at 1.0
                       while fitted and differene lines are scaled by same ratio.
        :param kwargs: kwargs for parameters in plotting. They are
            spacing: a float that is the spacing of data-difference and difference-data in data value.
                     Default is 0.
            apos: a tuple of two floats that are position of annotation in data frame.
                  Default is (.7, .3)
            rws: a list of Rw for each fitting. it only works when auto_rw = False. Defulat None.
            auto_rw: a bool to choose if the rw is automatically read and added to the annotation. Default is False.
            rwpos: a string to choose how rw is attached to the names. Default is "\n".
        :return: a matpotlib figure object.
    """
    # check args
    files = check_files(files)
    rlim = check_lim(rlim, 2)
    names = check_names(names, files)
    colors = check_colors(colors, len(files))

    # get kwargs
    options = ["spacing", "apos", "auto_rw", "rwpos", "rws"]
    check_kwargs(kwargs, options)
    spacing = kwargs.get("spacing", 0.)
    apos = kwargs.get("apos", (.7, .3))
    auto_rw = kwargs.get("auto_rw", False)
    rwpos = kwargs.get("rwpos", "\n")
    rws = kwargs.get("rws", None)

    # get rw
    if auto_rw:
        rws_from_file = get_rw(files)
        names = attach_rw(names, rws_from_file, rwpos)
    elif rws is not None:
        names = attach_rw(names, rws, rwpos)
    else:
        pass

    # load data
    rs, gcalcs, gs, gdiffs = load_fgr(files)

    # slice data
    if rlim:
        for n, (r, gcalc, g, gdiff) in enumerate(zip(rs, gcalcs, gs, gdiffs)):
            mask = np.logical_and(r >= rlim[0], r <= rlim[1])
            rs[n] = r[mask]
            gs[n] = g[mask]
            gcalcs[n] = gcalc[mask]
            gdiffs[n] = gdiff[mask]

    # normal
    if normal:
        normalize_fgr(gs, gcalcs, gdiffs)
    else:
        pass

    # offset gdiffs
    gzeros = offset_fgr(gs, gcalcs, gdiffs, spacing)

    # initiate figure
    fig = plt.figure()
    rct = (.1, .1, .8, .8)
    axs = add_axes(fig, rct, len(files))

    # initiate lines
    ldatas, lcalcs, ldiffs, lzeros = add_fgrlines(axs, rs, gs, gcalcs, gdiffs, gzeros)

    # add annotations
    poss = calc_eachposs(rs, gs, apos)
    anns = annotate_plots(axs, names, poss)

    # set names and colors
    if colors is not None:
        paint_color(anns, colors)
        paint_color(ldatas, colors)
        comp_colors = get_comp(colors)
        paint_color(lcalcs, comp_colors)
        paint_color(ldiffs, colors)
    else:
        pass

    # configure axes
    config_axes(axs, rlim, r"r ($\AA$)", r"G ($\AA^{-2}$)")

    return fig


# project specific functions
def plot_example(files: Union[str, List[str]],
                 rlim: Tuple[float, float] = None,
                 names: Union[str, List[str]] = None,
                 color0: str = None,
                 colors: Union[str, List[str]] = None,  **kwargs):
    """
    plot an example for multiphase model fitting for Randy's report.
    :param files: a list of files. The first file is .fgr file and the rest is multiple .gr files.
    :param rlim: a tuple of two floats that are endpoints of data.
    :param names: a list of strings that are annotations for lines.
                  If names is empty list, no names are added, elif names is None, names are file names.
    :param color0: color for the data.
    :param colors: a list of strings that are hex values starting with "#" for colors.
    :param kwargs: kwargs for plot settings. They are
           apos: a tuple of two floats as fractional posistion of annotation in data frame. Default is (.8, .5).
    :return: a matpotlib figure object.
    """
    # check arg
    files = check_files(files)
    rlim = check_lim(rlim, 2)
    names = check_names(names, files)
    color0 = check_colors(color0, 1)
    colors = check_colors(colors, len(files))
    # get kwargs
    apos = kwargs.get("apos", (.8, .3))
    # load data
    r0, g_calc0, g0, _ = load_fgr([files[0]])
    rs, g_calcs = load_gr(files[1:])
    rs = r0 + rs
    g_calcs = g_calc0 + g_calcs
    # initiate figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot data
    shift_data(g_calcs, spacing=.3)
    line0 = add_circlines(ax, r0, g0)
    lines = add_solidlines(ax, rs, g_calcs)
    # add annotations
    poss = calc_poss(rs, g_calcs, apos)
    anns = annotate_plots(ax, names, poss)
    # add colors
    if color0 is None:
        pass
    else:
        paint_color(line0, color0)
    if colors is not None:
        paint_color(lines, colors)
        paint_color(anns, colors)
    else:
        pass
    # configure axes
    config_ax(ax, rlim, None, r'r ($\AA$)', r'G ($\AA^{-2}$)')
    return fig
