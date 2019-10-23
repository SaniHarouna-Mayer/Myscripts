import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing
from typing import Tuple, Union, List
from scipy.optimize import least_squares
from uuid import uuid4
from diffpy.srfit.structure.sgconstraints import constrainAsSpaceGroup
import diffpy.srfit.pdf.characteristicfunctions as characteristicfunctions

from diffpy.structure import loadStructure
from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator, PDFParser
from diffpy.srfit.fitbase import Profile, FitContribution, FitResults
from diffpy.utils.parsers.loaddata import loadData
from myscripts.fittingclass import GenConfig, ConConfig, MyRecipe
from deprecated import deprecated


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


def make_generator(config: GenConfig) -> Union[PDFGenerator, DebyePDFGenerator]:
    """
    build generator from a Phase object.
    :param config: Phase object.
    :return:
        generator: generator object.
    """
    name = config.name
    stru = loadStructure(config.stru_file)
    ncpu = config.ncpu

    if config.debye:
        generator = DebyePDFGenerator(name)
    else:
        generator = PDFGenerator(name)

    generator.setStructure(stru, periodic=config.periodic)

    if ncpu:
        pool = multiprocessing.Pool(ncpu)
        generator.parallel(ncpu, mapfunc=pool.imap_unordered)
    else:
        pass

    return generator


def make_contribution(config: ConConfig) -> FitContribution:
    """
    make recipe for fitting, including the data and the model, without any fitting variables set or added.
    :param config: Model object.
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

    contribution = FitContribution(config.name)

    fit_range = config.fit_range
    profile = make_profile(config.data_file, fit_range)
    contribution.setProfile(profile, xname="r")

    _, qmax_from_data = read_qrange(config.data_file)
    for phase in config.phases:
        generator = make_generator(phase)
        generator.qdamp.value = config.qparams[0]
        generator.qbroad.value = config.qparams[1]
        set_qmin_qmax(generator)
        contribution.addProfileGenerator(generator)

    for base_line in config.base_lines:
        contribution.addProfileGenerator(base_line)

    for function in config.functions:
        name = function.name
        func_type = function.func_type
        argnames = function.argnames
        contribution.registerFunction(func_type, name, argnames)

    contribution.setEquation(config.eq)
    contribution.setResidualEquation(config.res_eq)

    return contribution


def make(*configs: ConConfig, weights: List[float] = None) -> MyRecipe:
    """
    make recipe based on models.
    :param configs: arbitrary number of model objects. each model make one contribution in recipe.
    :param weights: list of weights for each contribution. if None, weight will be 1. / len(models).
    :return: fit recipe.
    """
    recipe = MyRecipe(*configs)
    if weights is None:
        weights = [1. / len(configs)] * len(configs)
    else:
        msg = f"models and weights doe not have same length: {len(configs)}, {len(weights)}."
        assert len(configs) == len(weights), msg

    for model, weight in zip(configs, weights):
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


def _make_df(recipe: MyRecipe) -> pd.DataFrame:
    """
    get Rw and fitting parameter values from recipe and make them a pandas dataframe
    :param recipe: fit recipe.
    :return:
    """
    df = pd.DataFrame()
    res = recipe.res = FitResults(recipe)
    df["name"] = ["Rw", "half_chi2"] + res.varnames
    df["val"] = [res.rw, res.chi2 / 2] + res.varvals.tolist()
    df["std"] = [0, 0] + res.varunc
    df = df.set_index("name")
    return df


def fit(recipe: MyRecipe, **kwargs) -> None:
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
        df = _make_df(recipe)
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


@deprecated(version='1.0', reason="This function is deprecated.")
def save(recipe: MyRecipe, con_names: Union[str, List[str]], base_name: str) -> Tuple[str, Union[List[str], str]]:
    """
    save fitting result and fitted gr. the fitting result will be saved as csv file with name same as the file_name.
    the fitted gr will be saved with name of file_name followed by index of the contribution if there are multiple
    contributions.
    :param recipe: fit recipe.
    :param con_names: single or a list of names of fitcontribution.
    :param base_name: base name for the saving file.
    :return: path to saved csv file, path to saved fgr file or a list of the path to saved fgr files.
    """
    df = _make_df(recipe)
    csv_file = rf"{base_name}.csv"
    df.to_csv(csv_file)

    rw = df.iloc[0, 0]
    if isinstance(con_names, str):
        con = getattr(recipe, con_names)
        fgr = rf"{base_name}_{con_names}.fgr"
        con.profile.savetxt(fgr, header=f"Rw = {rw}\nx ycalc y dy")
    else:
        fgr = []
        for con_name in con_names:
            con: FitContribution = getattr(recipe, con_name)
            fgr_file = rf"{base_name}_{con_name}.fgr"
            fgr.append(fgr_file)
            con.profile.savetxt(fgr_file, header=f"Rw = {rw}\nx ycalc y dy")

    return csv_file, fgr


def save_csv(recipe: MyRecipe, base_name: str) -> str:
    """
    Save fitting results to a csv file.
    Parameters
    ----------
    recipe
        Fitrecipe.
    base_name
        base name for saving. The saving name will be "{base_name}.csv"

    Returns
    -------
    path to the csv file.
    """
    df = _make_df(recipe)
    csv_file = rf"{base_name}.csv"
    df.to_csv(csv_file)
    return csv_file


def save_fgr(contribution: FitContribution, base_name: str, rw: float) -> str:
    """
    Save fitted PDFs to a four columns txt files with Rw as header.
    Parameters
    ----------
    contribution
        arbitrary number of Fitcontributions.
    base_name
        base name for saving. The saving name will be "{base_name}_{contribution.name}.fgr"
    rw
        value of Rw. It will be in the header as "Rw = {rw}".

    Returns
    -------
        the path to the fgr file.
    """
    fgr_file = rf"{base_name}_{contribution.name}.fgr"
    contribution.profile.savetxt(fgr_file, header=f"Rw = {rw}\nx ycalc y dy")
    return fgr_file


def save_cif(generator: Union[PDFGenerator, DebyePDFGenerator], base_name: str, con_name: str, ext: str = "cif") -> str:
    """
    Save refined structure.
    Parameters
    ----------
    generator
        arbitrary number of generators.
    base_name
        base name for saving. The saving name will be "{base_name}_{con_name}_{generator.name}."
    con_name
        name of the contribution that the generators belong to.
    ext
        extension of the structure file. It will also determine the structure file type. Default "cif".
    Returns
    -------
        the path to the cif or xyz files.
    """
    stru_file = rf"{base_name}_{con_name}_{generator.name}.{ext}"
    generator.stru.write(stru_file, ext)
    return stru_file


def save_all(recipe: MyRecipe, folder: str, name: str, info: dict = None) -> str:
    """
    Save fitting results, fitted PDFs and refined structures to files in one folder and save information in DataFrames.
    Parameters
    ----------
    recipe
        Refined recipe to save.
    name
        Basic name of the saving files.
    folder
        Folder to save the files.
    info
        information to update in DataFame. Each key will be column and value will be the content of the cell.
    Returns
    -------
        string of Uid.
    """
    uid = str(uuid4())[:4]
    name += f"_{uid}"
    name = os.path.join(folder, name)

    csv_file = save_csv(recipe, name)
    csv_info = dict(file=csv_file, **info)
    recipe.csv_df = recipe.csv_df.append(csv_info, ignore_index=True)

    for config in recipe.configs:
        con = getattr(recipe, config.name)
        fgr_file = save_fgr(con, base_name=name, rw=recipe.res.rw)
        fgr_info = dict(file=fgr_file, rw=recipe.res.rw, **info)
        recipe.fgr_df =recipe.fgr_df.append(fgr_info, ignore_index=True)

        for gconfig in config.phases:
            gen = getattr(con, gconfig.name)
            cif_file = save_cif(gen, base_name=name, con_name=config.name)
            cif_info = dict(file=cif_file, rw=recipe.res.rw, **info)
            recipe.cif_df = recipe.cif_df.append(cif_info, ignore_index=True)

    return uid


def updated(recipe: MyRecipe, csv_df: pd.DataFrame = None, fgr_df: pd.DataFrame = None, cif_df: pd.DataFrame = None) ->\
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Update the information DataFrame using information stored in recipe.
    Parameters
    ----------
    recipe
        MyRecipe storing the information of saved files.
    csv_df
        Information DataFrame of csvs of fitting results.
    fgr_df
        Information DataFrame of fgrs of fitted PDFs.
    cif_df
        Information DataFrame of cifs of refined structures.
    Returns
    -------
    csv_df
        updated csv DataFrame
    fgr_df
        updated fgr DataFrame
    cif_df
        updated cif DataFrame
    """
    if csv_df is not None:
        csv_df = csv_df.append(recipe.csv_df)
    if fgr_df is not None:
        fgr_df = fgr_df.append(recipe.fgr_df)
    if cif_df is not None:
        cif_df = cif_df.append(recipe.cif_df)
    return csv_df, fgr_df, cif_df
