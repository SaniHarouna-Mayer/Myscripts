import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing
from typing import Tuple, Union, List, Dict
from scipy.optimize import least_squares
from uuid import uuid4
from datetime import datetime
from collections import Counter
from diffpy.structure import Structure
from diffpy.srfit.structure.sgconstraints import constrainAsSpaceGroup
import diffpy.srfit.pdf.characteristicfunctions as characteristicfunctions
from diffpy.structure import loadStructure
from diffpy.srfit.pdf import PDFGenerator, DebyePDFGenerator, PDFParser
from diffpy.srfit.fitbase import Profile, FitContribution, FitResults
from diffpy.utils.parsers.loaddata import loadData
from myscripts.fittingclass import GenConfig, ConConfig, MyRecipe


__all__ = ["make_profile", "make_generator", "make", "fit", "old_save", "gen_save_all", "F", "plot",
           "constrainAsSpaceGroup", "calc", "load_default", 'sgconstrain', 'free_and_fit']


# abbreviate some useful modules and functions
F = characteristicfunctions
constrainAsSpaceGroup = constrainAsSpaceGroup


# functions used in fitting
def make_profile(data_file: str, fit_range: Tuple[float, float, float]) -> Profile:
    """
    Make a Profile, parse data file to it and set its calculation range.

    Parameters
    ----------
    data_file
        The path to the data file.
    fit_range
        The tuple of (rmax, rmin, dr) in Angstrom.
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
        ConConfigs. If None, every FitContribution has the same weight 1.

    Returns
    -------
    recipe
        MyRecipe built from ConConfigs.

    """
    recipe = MyRecipe(configs=configs)
    if weights is None:
        weights = [1.] * len(configs)
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
    Calculate the value of generator and compare it with the data in the file.

    Parameters
    ----------
    gen
        The PDFGenerator or DebyePDFGenerator.
    data_file
        The path to the data file.
    rlim
        The limit of calculation range.

    Returns
    -------
    None

    """
    r, g_data = loadData(data_file).T
    msk = np.logical_and(r >= rlim[0], r <= rlim[1])
    r, g_data = r[msk], g_data[msk]

    g_calc = gen(r)

    plt.figure()
    plt.plot(r, g_data, label="data")
    plt.plot(r, g_calc, label="calculation")
    plt.xlabel(r"r ($\AA$)")
    plt.ylabel(r"G ($\AA^{-2}$)")
    plt.legend()
    plt.show()
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
        df, res = _make_df(recipe)
        print(f"Fitting Results of {recipe.name}")
        print("-" * 90)
        print(df.to_string())
        print("-" * 90)
        print("\n")
    else:
        pass

    return


def plot(recipe: MyRecipe) -> None:
    """
    Plot the fits for all FitContributions in the recipe.

    Parameters
    ----------
    recipe
        The FitRecipe.

    Returns
    -------
    None

    """
    for config in recipe.configs:
        contribution = getattr(recipe, config.name)

        r = contribution.profile.x
        g = contribution.profile.y
        gcalc = contribution.profile.ycalc
        diff = g - gcalc
        offset = min([g.min(), gcalc.min()]) - diff.max()
        diffzero = offset * np.ones_like(diff)
        diff += diffzero

        plt.figure()
        plt.title(config.name)
        plt.plot(r, g, 'bo', mfc="None", label="Data")
        plt.plot(r, gcalc, 'r-', label="Calculation")
        plt.plot(r, diff, 'g-', label="Difference")
        plt.plot(r, diffzero, 'k-')
        plt.xlabel(r"$r (\AA)$")
        plt.ylabel(r"$G (\AA^{-2})$")
        plt.legend(loc=1)
        plt.show()
    return


def old_save(recipe: MyRecipe, con_names: Union[str, List[str]], base_name: str) -> Tuple[str, Union[List[str], str]]:
    """
    save fitting result and fitted gr. the fitting result will be saved as csv file with name same as the file_name.
    the fitted gr will be saved with name of file_name followed by index of the contribution if there are multiple
    contributions.
    :param recipe: fit recipe.
    :param con_names: single or a list of names of fitcontribution.
    :param base_name: base name for the saving file.
    :return: path to saved csv file, path to saved fgr file or a list of the path to saved fgr files.
    """
    df, res = _make_df(recipe)
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


def save_csv(recipe: MyRecipe, base_name: str) -> Tuple[str, float, float]:
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
    df, res = _make_df(recipe)
    csv_file = rf"{base_name}.csv"
    df.to_csv(csv_file)
    rw = res.rw
    half_chi2 = res.chi2 / 2.
    return csv_file, rw, half_chi2


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


def gen_save_all(folder: str, csv: str, fgr: str, cif: str):
    """
    Generate the function save_all to save results of recipes. The database of csv, fgr and cif will be passed to the
    "_save_all" function. If there is no such file, it will be created as an empty csv file.

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
    for filepath in (csv, fgr, cif):
        if not os.path.isfile(filepath):
            pd.DataFrame().to_csv(filepath)

    def save_all(recipe: MyRecipe, tag: str = None):
        """
        Save fitting results, fitted PDFs and refined structures to files in one folder and save information in
        DataFrames. The DataFrame will contain columns: 'file' (file paths), 'rw' (Rw value) and other information in
        info.

        Parameters
        ----------
        recipe
            The FitRecipe.
        tag
            A tag to add in csv database.

        Returns
        -------
        uid
            The uid of the saving.
        """
        return _save_all(recipe, folder, csv, fgr, cif, tag)

    return save_all


def _save_all(recipe: MyRecipe, folder: str, csv: str, fgr: str, cif: str, tag: str = None) -> str:
    """
    Save fitting results, fitted PDFs and refined structures to files in one folder and save information in DataFrames.
    The DataFrame will contain columns: 'file' (file paths), 'rw' (Rw value) and other information in info.

    Parameters
    ----------
    recipe
        Refined recipe to save.
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
        string of Uid.
    """
    print(f"Save {recipe.name}...\n")
    uid = str(uuid4())[:4]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = os.path.join(folder, f"{timestamp}_{uid}")

    csv_file, rw, half_chi2 = save_csv(recipe, name)
    csv_info = dict(csv_file=csv_file, rw=rw, half_chi2=half_chi2, timestamp=timestamp, tag=tag)
    recipe_id = update(csv, csv_info, id_col='recipe_id')

    for config in recipe.configs:
        con = getattr(recipe, config.name)
        fgr_file = save_fgr(con, base_name=name, rw=rw)
        config_info = config.to_dict()
        fgr_info = dict(recipe_id=recipe_id, fgr_file=fgr_file, **config_info)
        con_id = update(fgr, fgr_info, id_col='con_id')

        for gconfig in config.phases:
            gen = getattr(con, gconfig.name)
            cif_file = save_cif(gen, base_name=name, con_name=config.name)
            gconfig_info = gconfig.to_dict()
            cif_info = dict(con_id=con_id, recipe_id=recipe_id, cif_file=cif_file, **gconfig_info)
            update(cif, cif_info, id_col='gen_id')

    return uid


def update(file_path: str, info_dct: dict, id_col: str) -> int:
    """
    Update the database file (a csv file) by appending the information as a row at the end of the dataframe and return
    a serial id of for the piece of information.

    Parameters
    ----------
    file_path
        The path to the csv file that stores the information.
    info_dct
        The dictionary of information.
    id_col
        The column name of the id.

    Returns
    -------
    id_val
        An id for the information.

    """
    df = pd.read_csv(file_path)
    row_dct = {id_col: df.shape[0]}
    row_dct.update(**info_dct)
    if df.empty:
        newdf = pd.DataFrame([row_dct])
    else:
        newdf = df.append(row_dct, ignore_index=True, sort=False)
    newdf.to_csv(file_path, index=False)
    return row_dct[id_col]


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


def sgconstrain(recipe: MyRecipe, gen_name: str, sg: Union[int, str] = None,
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
    gen_name
        The name of the generator to constrain. It assumes the generators in the recipe have unique name.
    sg
        The space group. The expression can be the string or integer. If None, use the space group in GenConfig.
    dv
        The default value of the constrained parameters. If None, the default values will be used.
    scatterers
        The argument scatters of the constrainAsSpaceGroup. If None, None will be used.

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


def free_and_fit(recipe: MyRecipe, *tags: Union[str, Tuple[str]], **kwargs) -> None:
    """
    First fix all variables and then free the variables one by one and fit the recipe.

    Parameters
    ----------
    recipe
        The recipe to fit.
    tags
        The tags of variables to free. It can be single tag or a tuple of tags.
    kwargs
        The kwargs of the 'fit'.

    Returns
    -------
    None

    """
    print(f"Start {recipe.name} with all parameters fixed:")
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
        if n == len(tags) - 1 and '_print' not in kwargs:
            fit(recipe, _print=True, **kwargs)
        else:
            fit(recipe, **kwargs)
    return
