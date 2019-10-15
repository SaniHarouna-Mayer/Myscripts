import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import pandas as pd
from diffpy.structure import loadStructure
from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator, PDFParser
from diffpy.srfit.fitbase import Profile, FitContribution, FitRecipe, FitResults
from diffpy.utils.parsers.loaddata import loadData
from myscripts.fittingclass import *
from typing import Tuple
import os
from diffpy.srfit.structure.sgconstraints import constrainAsSpaceGroup
import diffpy.srfit.pdf.characteristicfunctions as characteristicfunctions
import multiprocessing


# abbreviate some useful modules and functions
F = characteristicfunctions
constrainAsSpaceGroup = constrainAsSpaceGroup


# functions used in fitting
def make_profile(data_file: str, fit_range: Tuple[float, float, float]) -> Profile:
    """
    build profile for contribution.
    :param data_file: name of the data file.
    :param fit_range: fit range: (rmin, rmax, rstep).
    :return:
        profile: profile object.
    """
    profile = Profile()
    parser = PDFParser()
    parser.parseFile(data_file)
    profile.loadParsedData(parser)

    rmin, rmax, rstep = fit_range
    profile.setCalculationRange(rmin, rmax, rstep)

    return profile


def make_generator(phase: Phase) -> Union[PDFGenerator, DebyePDFGenerator]:
    """
    build generator from a Phase object.
    :param phase: Phase object.
    :return:
        generator: generator object.
    """
    name = phase.name
    stru = loadStructure(phase.stru_file)
    ncpu = phase.ncpu

    if phase.debye:
        generator = DebyePDFGenerator(name)
    else:
        generator = PDFGenerator(name)

    generator.setStructure(stru, periodic=phase.periodic)

    if ncpu:
        pool = multiprocessing.Pool(ncpu)
        generator.parallel(ncpu, mapfunc=pool.imap_unordered)
    else:
        pass

    return generator


def make_contribution(model: Model) -> FitContribution:
    """
    make recipe for fitting, including the data and the model, without any fitting variables set or added.
    :param model: Model object.
    :return:
    """
    def read_qrange(_data_file: str):
        _qmin, _qmax = None, None
        with open(_data_file, "r") as f:
            while True:
                line = f.readline()
                value_found = _qmin is not None and _qmax is not None
                end_of_file = len(line) == 0
                if value_found or end_of_file:
                    break
                elif "qmin = " in line:
                    _qmin = float(line.split()[2])
                elif "qmax = " in line:
                    _qmax = float(line.split()[2])
                else:
                    pass
        if value_found:
            pass
        else:
            message = "values of qmin and qmax are not found:" + \
                      f"qmin:{_qmin}; qmax{_qmax}"
            raise Exception(message)
        return _qmin, _qmax

    def choose_qmin(gen: Union[DebyePDFGenerator, PDFGenerator]):
        if isinstance(gen, DebyePDFGenerator):
            _qmin = 1.
        elif isinstance(gen, PDFGenerator):
            _qmin = 0.
        else:
            raise Exception(f"Unknown generator {type(generator)}.")
        return _qmin

    def set_qmin_qmax(_generator):
        if phase.qmin is None:
            qmin = choose_qmin(_generator)
        else:
            qmin = phase.qmin

        if phase.qmax is None:
            qmax = qmax_from_data
        else:
            qmax = phase.qmax

        _generator.setQmin(qmin)
        _generator.setQmax(qmax)
        return

    contribution = FitContribution(model.name)

    fit_range = model.fit_range
    profile = make_profile(model.data_file, fit_range)
    contribution.setProfile(profile, xname="r")

    _, qmax_from_data = read_qrange(model.data_file)
    for phase in model.phases:
        generator = make_generator(phase)
        generator.qdamp.value = model.qparams[0]
        generator.qbroad.value = model.qparams[1]
        set_qmin_qmax(generator)
        contribution.addProfileGenerator(generator)

    for base_line in model.base_lines:
        contribution.addProfileGenerator(base_line)

    for function in model.functions:
        name = function.name
        func_type = function.func_type
        argnames = function.argnames
        contribution.registerFunction(func_type, name, argnames)

    contribution.setEquation(model.eq)
    contribution.setResidualEquation(model.res_eq)

    return contribution


def make(*models: Model, weights: List = None) -> FitRecipe:
    """
    make recipe based on models.
    :param models: arbitrary number of model objects. each model make one contribution in recipe.
    :param weights: list of weights for each contribution. if None, weight will be 1. / len(models).
    :return: fit recipe.
    """
    recipe = FitRecipe()
    if weights is None:
        weights = [1. / len(models)] * len(models)
    else:
        msg = f"models and weights doe not have same length: {len(models)}, {len(weights)}."
        assert len(models) == len(weights), msg
        pass

    for model, weight in zip(models, weights):
        contribution = make_contribution(model)
        recipe.addContribution(contribution, weight)

    recipe.fithooks[0].verbose = 0

    return recipe


def calc(gen: Union[PDFGenerator, DebyePDFGenerator],
         data_file: str,
         rlim: Tuple[float, float]) -> None:
    """
    calculate PDF according to generator in recipe and compare with the data.
    :param gen: generator used to calculate PDF.
    :param data_file: path to the data file.
    :param rlim: limit of r-range.
    :return: None.
    """
    r, g_data = loadData(data_file).T
    msk = np.logical_and(r >= rlim[0], r <= rlim[1])
    r, g_data = r[msk], g_data[msk]

    g_calc = gen(r)

    plt.plot(r, g_data, label="data")
    plt.plot(r, g_calc, label="calculation")

    plt.xlabel(r"r ($\AA$)")
    plt.ylabel(r"G ($\AA^{-2}$)")

    plt.legend()

    return


def make_df(recipe: FitRecipe, name: str = "") -> pd.DataFrame:
    """
    get Rw and fitting parameter values from recipe and make them a pandas dataframe
    :param recipe: fit recipe.
    :param name: name of the column. Default ""
    :return:
    """
    df = pd.DataFrame()
    res = FitResults(recipe)
    df[name] = [res.rw, res.chi2/2] + list(recipe.getValues())
    df.index = ["Rw", "half_chi2"] + recipe.getNames()
    return df


def fit(recipe: FitRecipe, **kwargs) -> None:
    """
    fit the data according to recipe. parameters associated with fitting can be set in kwargs.
    :param recipe: recipe to fit.
    :param kwargs: parameters in fitting. including
                   verbose: how much information to print. Default 2
                   values: initial value for fitting. Default get from recipe
                   bounds: two list of lower and upper bounds. Default get from recipe
                   xtol, gtol, ftol: tolerance in least squares. Default 1.E-4, 1.E-4, 1.E-4
                   max_nfev: maximum number of evaluation of residual function. Default None
                   _print: whether to print the data. Default False
    :return:
    """
    values = kwargs.get("values", recipe.values)
    bounds = kwargs.get("bounds", recipe.getBounds2())
    verbose = kwargs.get("verbose", 2)
    xtol = kwargs.get("xtol", 1.E-4)
    gtol = kwargs.get("gtol", 1.E-4)
    ftol = kwargs.get("ftol", 1.E-4)
    max_nfev = kwargs.get("max_fev", None)
    least_squares(recipe.residual, values, bounds=bounds, verbose=verbose, xtol=xtol, gtol=gtol, ftol=ftol,
                  max_nfev=max_nfev)

    _print = kwargs.get("_print", False)
    if _print:
        df = make_df(recipe)
        print("-" * 90)
        print(df.to_string())
        print("-" * 90)
    else:
        pass

    return


def plot(contribution: FitContribution, ax: plt.Axes = None) -> None:
    """
    plot result of fitting.
    :param contribution: fit contribution to plot.
    :param ax: axes to plot. Default None. New axes and figure are created.
    :return:
    """
    # All this should be pretty familiar by now.
    r = contribution.profile.x
    g = contribution.profile.y
    gcalc = contribution.profile.ycalc
    diff = g - gcalc
    offset = min([g.min(), gcalc.min()]) - diff.max()
    diffzero = offset * np.ones_like(diff)
    diff += diffzero

    if ax is None:
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot(111)
    else:
        pass
    ax.plot(r, g, 'bo', label="Data")
    ax.plot(r, gcalc, 'r-', label="Fit")
    ax.plot(r, diff, 'g-', label="Difference")
    ax.plot(r, diffzero, 'k-')
    ax.set_xlabel(r"$r (\AA)$")
    ax.set_ylabel(r"$G (\AA^{-2})$")
    ax.legend(loc=1)

    return


def save(recipe: FitRecipe, con_names: Union[str, List[str]], base_name: str) -> Tuple[str, str]:
    """
    save fitting result and fitted gr. the fitting result will be saved as csv file with name same as the file_name.
    the fitted gr will be saved with name of file_name followed by index of the contribution if there are multiple
    contributions.
    :param recipe: fit recipe.
    :param con_names: single or a list of names of fitcontribution.
    :param base_name: base name for the saving file.
    :return: path to saved csv file, path to saved fgr file or a list of the path to saved fgr files.
    """
    def choose_file_name():
        n = 0
        file_path = f"{base_name}_{n}"
        while os.path.isfile(file_path+".csv") or os.path.isfile(file_path+".fgr"):
            n += 1
            file_path = f"{base_name}_{n}"
        return file_path

    base = choose_file_name()

    df = make_df(recipe, name="val")
    csv_file = base + ".csv"
    df.to_csv(csv_file)

    rw = df.iloc[0, 0]
    if isinstance(con_names, str):
        con = getattr(recipe, con_names)
        fgr_files = base + ".fgr"
        con.profile.savetxt(base + ".fgr", header=f"Rw = {rw}\nx ycalc y dy")
    else:
        fgr_files = []
        for con_name in con_names:
            con: FitContribution = getattr(recipe, con_name)
            fgr_file = base + f"_{con_name}.fgr"
            fgr_files.append(fgr_file)
            con.profile.savetxt(fgr_file, header=f"Rw = {rw}\nx ycalc y dy")

    return csv_file, fgr_files
