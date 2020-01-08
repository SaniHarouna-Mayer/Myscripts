import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing
from typing import Tuple, Union, List, Dict
from scipy.optimize import least_squares
from uuid import uuid4
from collections import Counter
from diffpy.structure import Structure
from diffpy.srfit.structure.sgconstraints import constrainAsSpaceGroup
import diffpy.srfit.pdf.characteristicfunctions as characteristicfunctions
from diffpy.structure import loadStructure
from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator, PDFParser
from diffpy.srfit.fitbase import Profile, FitContribution, FitResults
from diffpy.utils.parsers.loaddata import loadData
from myscripts.fittingclass import GenConfig, ConConfig, MyRecipe


__all__ = ["make_profile", "make_generator", "make", "fit", "gen_save_all", "save", "F", "plot",
           "constrainAsSpaceGroup", "calc", "load_default", 'sgconstrain', 'free_and_fit']


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
    Build a generator according to the information in the GenConfig.

    Parameters
    ----------
    config : GenConfig
        A configuration instance for generator building.

    Returns
    -------
    generator: PDFGenerator or DebyePDFGenerator
        A generator built from GenConfig.

    """
    name = config.name
    stru: Structure = loadStructure(config.stru_file)
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
    Make a FitContribution according to the ConConfig.

    Parameters
    ----------
    config : ConConfig
        The configuration instance for the FitContribution.

    Returns
    -------
    contribution : FitContribution
        The FitContribution built from ConConfig.

    """
    contribution = FitContribution(config.name)

    fit_range = config.fit_range
    profile = make_profile(config.data_file, fit_range)
    contribution.setProfile(profile, xname="r")

    for phase in config.phases:
        generator = make_generator(phase)
        generator.qdamp.value = config.qparams[0]
        generator.qbroad.value = config.qparams[1]
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
    Make a FitRecipe based on single or multiple ConConfig.

    Parameters
    ----------
    configs
        The configurations of single or multiple FitContribution.
    weights
        The weights for the evaluation of each FitContribution. It should have the same length as the number of
        ConConfigs.

    Returns
    -------
    recipe
        MyRecipe built from ConConfigs.

    """
    recipe = MyRecipe(configs=configs)
    if weights is None:
        weights = [1. / len(configs)] * len(configs)
    else:
        msg = f"models and weights doe not have same length: {len(configs)}, {len(weights)}."
        assert len(configs) == len(weights), msg

    for config, weight in zip(configs, weights):
        contribution = make_contribution(config)
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


def _make_df(recipe: MyRecipe) -> Tuple[pd.DataFrame, FitResults]:
    """

    :param recipe: fit recipe.
    :return:
    """
    df = pd.DataFrame()
    res = FitResults(recipe)
    df["name"] = ["Rw", "half_chi2"] + res.varnames
    df["val"] = [res.rw, res.chi2 / 2] + res.varvals.tolist()
    df["std"] = [np.nan, np.nan] + res.varunc
    df = df.set_index("name")
    return df, res


def fit(recipe: MyRecipe, **kwargs) -> None:
    """
    Fit the data according to recipe. parameters associated with fitting can be set in kwargs.

    Parameters
    ----------
    recipe
        MyRecipe to fit.
    kwargs
        Parameters in fitting. They are
            verbose: how much information to print. Default 2.
            values: initial value for fitting. Default get from recipe.
            bounds: two list of lower and upper bounds. Default get from recipe.
            xtol, gtol, ftol: tolerance in least squares. Default 1.E-4, 1.E-4, 1.E-4.
            max_nfev: maximum number of evaluation of residual function. Default None.
            _print: whether to print the data. Default False.
    Returns
    -------
    None

    """
    values = kwargs.get("values", recipe.values)
    bounds = kwargs.get("bounds", recipe.getBounds2())
    verbose = kwargs.get("verbose", 2)
    xtol = kwargs.get("xtol", 1.E-8)
    gtol = kwargs.get("gtol", 1.E-4)
    ftol = kwargs.get("ftol", 1.E-4)
    max_nfev = kwargs.get("max_fev", None)
    least_squares(recipe.residual, values, bounds=bounds, verbose=verbose, xtol=xtol, gtol=gtol, ftol=ftol,
                  max_nfev=max_nfev)

    _print = kwargs.get("_print", False)
    if _print:
        df = _make_df(recipe)
        print(f"Fitting Results of {recipe.name}")
        print("-" * 90)
        print(df.to_string())
        print("-" * 90)
        print("\n")
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
    ax.plot(r, g, 'bo', mfc="None", label="Data")
    ax.plot(r, gcalc, 'r-', label="Fit")
    ax.plot(r, diff, 'g-', label="Difference")
    ax.plot(r, diffzero, 'k-')
    ax.set_xlabel(r"$r (\AA)$")
    ax.set_ylabel(r"$G (\AA^{-2})$")
    ax.legend(loc=1)

    return


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


def gen_save_all(folder: str, csv: str = None, fgr: str = None, cif: str = None):
    """
    Generate the function save_all to save results of recipes.

    Parameters
    ----------
    folder
            folder
        Folder to save the files.
    csv
        The path to the csv file containing fitting results information.
    fgr
        The path to the csv file containing fitted PDFs information.
    cif
        The path to the csv file containing refined structure information.

    Returns
    -------
    save_all
        A function to save results.

    """
    def save_all(recipe: MyRecipe, name: str = None, **kwargs):
        """
        Save fitting results, fitted PDFs and refined structures to files in one folder and save information in
        DataFrames. The DataFrame will contain columns: 'file' (file paths), 'rw' (Rw value) and other information in
        info.

        Parameters
        ----------
        recipe
            The FitRecipe.
        name
            The name of saving files.
        kwargs
            information to update in DataFame. Each key will be column and each value will be the content of the cell.

        Returns
        -------
        uid
            The uid of the saving.
        """
        return _save_all(recipe, folder, name, csv, fgr, cif, **kwargs)

    return save_all


def _save_all(recipe: MyRecipe, folder: str, name: str = None, csv: str = None, fgr: str = None, cif: str = None,
              **kwargs) -> str:
    """
    Save fitting results, fitted PDFs and refined structures to files in one folder and save information in DataFrames.
    The DataFrame will contain columns: 'file' (file paths), 'rw' (Rw value) and other information in info.

    Parameters
    ----------
    recipe
        Refined recipe to save.
    folder
        Folder to save the files.
    name
        Basic name of the saving files. If None, use recipe.name.
    csv
        The path to the csv file containing fitting results information.
    fgr
        The path to the csv file containing fitted PDFs information.
    cif
        The path to the csv file containing refined structure information.
    kwargs
        information to update in DataFame. Each key will be column and each value will be the content of the cell.

    Returns
    -------
        string of Uid.
    """
    print(f"Saving files of results from {recipe.name}...\n")
    uid = str(uuid4())[:8]
    name = f"{name}_{uid}" if name else f"{recipe.name}_{uid}"
    name = os.path.join(folder, name)

    csv_file = save_csv(recipe, name)
    csv_info = dict(uid=uid, file=csv_file, rw=recipe.res.rw, name=recipe.name, **kwargs)
    if csv:
        update(csv=csv, csv_info=csv_info)
    else:
        pass

    for config in recipe.configs:
        con = getattr(recipe, config.name)
        fgr_file = save_fgr(con, base_name=name, rw=recipe.res.rw)
        fgr_info = dict(uid=uid, file=fgr_file, name=recipe.name, rw=recipe.res.rw, **kwargs)
        if fgr:
            update(fgr=fgr, fgr_info=fgr_info)
        else:
            pass

        for gconfig in config.phases:
            gen = getattr(con, gconfig.name)
            cif_file = save_cif(gen, base_name=name, con_name=config.name)
            cif_info = dict(uid=uid, file=cif_file, name=recipe.name, rw=recipe.res.rw, **kwargs)
            if cif:
                update(cif=cif, cif_info=cif_info)
            else:
                pass

    return uid


def update(csv_info: dict = None, fgr_info: dict = None, cif_info: dict = None, csv: str = None,
           fgr: str = None, cif: str = None) -> None:
    """
    Update the information DataFrame using information stored in recipe.

    Parameters
    ----------
    csv_info
        Information of the csv file.
    fgr_info
        Information of the fgr file.
    cif_info
        Information of the cif file.
    csv
        The path to the csv file containing fitting results information.
    fgr
        The path to the csv file containing fitted PDFs information.
    cif
        The path to the csv file containing refined structure information.

    Returns
    -------
    None
    """
    file_infos = (csv_info, fgr_info, cif_info)
    dbs = (csv, fgr, cif)
    names = ("csv", "fgr", "cif")
    for file_info, db, name in zip(file_infos, dbs, names):
        if file_info:
            if db:
                df = pd.read_csv(db)
                df = df.append(file_info, ignore_index=True)
                df.to_csv(db, index=False)
            else:
                Warning(f"The information cannot be saved because '{db}' is None.")
        else:
            continue
    return


def load_default(csv_file: str):
    """
    Load the default value as a dictionary from the csv file of fitting results.

    Parameters
    ----------
    csv_file
        The path to the csv file.

    Returns
    -------
    default_val_dict
        A dictionary of variable names and its default values.
    """
    default_val_dict = pd.read_csv(csv_file, index_col=0)['val'].to_dict()
    return default_val_dict


def sgconstrain(recipe: MyRecipe, gen: Union[PDFGenerator, DebyePDFGenerator], sg: Union[int, str],
                dv: Dict[str, float] = None, scatterers: List = None) -> None:
    """
    Constrain the generator by space group. The constrained parameters are scale, delta2, lattice parameters, ADPs and
    xyz coordinates. The lattice constants and xyz coordinates are constrained by space group while the ADPs are
    constrained by elements. All paramters will be added as '{par.name}_{gen.name}'

    The default values, ranges and tags for parameters:
        scale: 0, (0, inf), scale_{gen.name}
        delta2: 0, (0, inf), delta2_{gen.name}
        lat: par.value, (par.value +/- 20%), lat_{gen.name}
        adp: 0.006, (0, inf), adp_{gen.name}
        xyz: par.value, (par.value +/- 0.2), xyz_{gen.name}

    Parameters
    ----------
    recipe
        The recipe to add variables.
    gen
        The generator to constrain.
    sg
        The space group. The expression can be the string or integer.
    dv
        The default value of the constrained parameters.
    scatterers
        The argument scatters of the constrainAsSpaceGroup.

    Returns
    -------
    None

    """
    dv = dv if dv else {}
    # add scale
    name = f'scale_{gen.name}'
    recipe.addVar(gen.scale, name=name, value=dv.get(name, 0.)).boundRange(0., np.inf)
    # add delta2
    name = f'delta2_{gen.name}'
    recipe.addVar(gen.delta2, name=name, value=dv.get(name, 0.)).boundRange(0., np.inf)

    # constrain lat
    sgpars = constrainAsSpaceGroup(gen.phase, sg, constrainadps=False, scatterers=scatterers)
    for par in sgpars.latpars:
        name = f'{par.name}_{gen.name}'
        tag = f'lat_{gen.name}'
        recipe.addVar(par, name=name, value=dv.get(name, par.value), tag=tag).boundWindow(par.value * 0.2)

    # constrain adp
    atoms = gen.phase.getScatterers()
    elements = Counter([atom.element for atom in atoms]).keys()
    adp = {element: recipe.newVar(f'Uiso_{element}_{gen.name}',
                                  value=dv.get(f'Uiso_{element}_{gen.name}', 0.006),
                                  tag=f'adp_{gen.name}').boundRange(0., np.inf)
           for element in elements}
    for atom in atoms:
        recipe.constrain(atom.Uiso, adp[atom.element])

    # constrain xyz
    for par in sgpars.xyzpars:
        name = f'{par.name}_{gen.name}'
        tag = f'xyz_{gen.name}'
        recipe.addVar(par, name=name, value=dv.get(name, par.value), tag=tag).boundWindow(0.2)

    return


def free_and_fit(recipe: MyRecipe, *tags, **kwargs) -> None:
    """
    First fix all variables and then free the variables one by one and fit the recipe.

    Parameters
    ----------
    recipe
        The recipe to fit.
    tags
        The tags of variables to free. It can be single string or a tuple of strings.
    kwargs
        The kwargs of the 'fit'.

    Returns
    -------
    None

    """
    print(f"Fit '{recipe.name}': ")
    recipe.fix('all')
    for n, tag in enumerate(tags):
        if isinstance(tag, tuple):
            print("Free " + ', '.join(tag) + ' ...')
            recipe.free(*tag)
        elif isinstance(tag, str):
            print(f"Free {tag} ...")
            recipe.free(tag)
        else:
            raise TypeError(f"Unknown tag type: {type(tag)}")
        if n == len(tags) - 1 and not '_print' in kwargs:
            fit(recipe, _print=True, **kwargs)
        else:
            fit(recipe, **kwargs)
    return
