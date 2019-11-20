from diffpy.pdfgetx import loaddata
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import numpy as np


def load(data_file):
    """
    load data from data file as two numpy array.
    :param data_file: (str) path to data file.
    :return: (Tuple[array, array]) x array and y array.
    """
    x, y = loaddata(data_file).T
    return x, y


def extract(data, x0, pdfgetter=None, rlim=None, out_file=None, options=None):
    """
    Extract the intermolecular signal data between molecule a and molecule b in their mixture from the PDF of mixture
    and single phase molecule a and molecule b. The intermolecular signal data is extracted by subtracting a linear
    combination of the PDFs of molecule a and molecule b from the PDF of mixture.
    G_inter(r) = G_ab(r) - [x_a * G_a(r) + x_b * G_b(r)]
    The coefficient in the linear combination is determined by minimizing the two-norm of G_inter data array after it
    is sliced according to rlim, which means the minimization is done to the PDF in range limited by rlim.
    The result will be plotted in two graphs, one graph of mixture PDF and linear combination result and their
    difference, the other graph of individual difference curve.
    Result of intermolecular signal will be save into a txt file if out_file is not None.
    :param data: (Tuple[Tuple[array, array], Tuple[array, array], Tuple[array, array]]) r array and g array pairs of 
    mol a + b, mol a and mol b (order matters).
    :param pdfgetter: (PDFGetter) a pdfgetter to process the data. If None, there will be no process to data.
    Default None.
    :param x0: (Tuple[float, float]) Initial guess of the proportional of PDF of molecule a. Default (0.5, 0.5).
    :param rlim: (Tuple[float, float]) lower and higher limit of the range of r for least square. If None, the whole
    range of data will be used. Unit A. Default None.
    :param out_file: (str) path to the output file of extracted intermolecular signal. If None, the intermolecular
    signal data won't be saved to files. Default None.
    :param options: (dict) keyword arguments passed to scipy.optimize.least_square to change the options of
    regression. If None, the default setting of least square will be used except the bounds is set to (0, inf).
    Default None.
    :return: (Tuple[numpy.array, numpy.array, numpy.array numpy.array]) array of r values, array of data G values, array
    of calculated G values, array of intermolecular signal.
    """
    # if no pdfgetter, create one pdfgetter that does nothing.
    if pdfgetter is None:
        from diffpy.pdfgetx import PDFGetter
        pdfgetter = PDFGetter()
        pdfgetter.transformations = []

    # process data and record results in two list
    rs = []  # rs: a list of numpy array of r value
    gs = []  # gs: a list of numpy array of g value
    from diffpy.pdfgetx import loaddata
    for x, y in data:
        r, g = pdfgetter(x, y)
        rs.append(r)
        gs.append(g)

    # window the data in place
    if rlim is not None:
        sliced_gs = []
        sliced_rs = []
        for r, g in zip(rs, gs):
            msk = np.logical_and(r >= rlim[0], r <= rlim[1])
            sliced_r = r[msk]
            sliced_g = g[msk]
            sliced_gs.append(sliced_g)
            sliced_rs.append(sliced_r)
    else:
        sliced_gs = gs
        sliced_rs = rs

    target = sliced_gs[0]
    component = np.array(sliced_gs[1:])

    # get coefficient using least square
    def residual(x: np.array):
        return target - np.dot(x, component)

    if options is None:
        options = {}
    res = least_squares(residual, x0, bounds=(0., np.inf), **options)
    x_opt = res.x  # optimized x value
    print(f"optimized x: {x_opt}")

    # calculate intermolecular PDF
    r = sliced_rs[0]
    g_data = sliced_gs[0]
    g_calc = np.dot(x_opt, sliced_gs[1:])
    g_inter = g_data - g_calc

    # save data to file
    if out_file is not None:
        header = f"optimized x: {x_opt}\n" + \
                 f"rlim: {rlim}" + \
                 "r g"
        np.savetxt(out_file, np.column_stack([r, g_data, g_calc, g_inter]), fmt="%.8f", header=header)

    return r, g_data, g_calc, g_inter


def plot(res):
    """
    plot the data PDF, calculated PDF and their difference (intermolecular signal) as a function of r. also, plot the
    difference as a function of r individually.
    :param res: (Tuple[r, g_data, g_calc, g_inter])
    r: (numpy.array) 1d array of r value.
    g_data: (numpy.array) 1d array of data G value, same shape as r.
    g_calc: (numpy.array) 1d array of calculated G value, same shape as r.
    g_inter: (numpy.array) 1d array of calculated G value, same shape as r.
    :return: None.
    """
    r, g_data, g_calc, g_inter = res
    # plot residual data
    plt.figure()
    plt.plot(r, g_inter, "-", label="difference")
    plt.xlabel(r"r ($\AA$)")
    plt.ylabel(r"G ($\AA^{-2}$)")
    plt.legend(loc=1)

    # plot data and calculation with difference curve
    plt.figure()
    plt.plot(r, g_data, "-", label="data")
    plt.plot(r, g_calc, "-", label="calculation")
    plt.plot(r, g_inter - 0.8 * np.max(g_data), "-", label="difference")
    plt.xlabel(r"r ($\AA$)")
    plt.ylabel(r"G ($\AA^{-2}$)")
    plt.legend(loc=1)
    plt.show()

    return
