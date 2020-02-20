import os
import json
import matplotlib.pyplot as plt
from diffpy.srfit.pdf import PDFGenerator
from diffpy.srfit.fitbase import FitContribution, FitResults
from diffpy.pdfgetx import loaddata, PDFGetter, PDFConfig
from myscripts.myint import xpdtools_int
from myscripts.fittingfunction import *
from myscripts.fittingclass import *
from myscripts.yamlmaker import load
from myscripts.helper import recfind
from typing import Callable, Tuple, Dict

__all__ = ["calib_wl", "run_pipe", "summarize", "plot_rw_wl", "PDFGETTER_CONFIG", "REFINE_CONFIG"]

XPDTOOLS_CONFIG = {
    "alpha": 2.0
}

PDFGETTER_CONFIG = {
    "composition": "Ni",
    "dataformat": "QA",
    "qmin": 0.,
    "qmax": 24.,
    "qmaxinst": 25.,
    "rmin": 0.,
    "rmax": 60.,
    "rstep": 0.01,
    "rpoly": 2.0
}

REFINE_CONFIG = {
    "fit_range": (1.5, 60., 0.01),
    "stru_file": os.path.join(os.path.dirname(__file__), "data_files", "Ni.cif"),
    "sgnum": 225,
    "qparams": (0.02, 0.04),
    "scale": 0.04,
    "delta2": 2.0,
    "uiso": 0.006
}


def calib_wl(tiff_file: str,
             working_dir: str,
             json_file: str = "result.json",
             pdfgetter: PDFGetter = None,
             refine_func: Callable = None) -> Dict[float, dict]:
    """
    Calibrate wavelength by integrating data in tiff_file, transforming to G(r) and fit with calibrant structure.
    The results will be dumped to a json file. It can be visualized by "summarize".

    Parameters
    ----------
    tiff_file
        Path to the tiff file.
    working_dir
        Path to the working directory, which hosts multiple child directories, each of which contains a poni
        file for a wavelength.
    json_file
        The file name of the json file of the results. It will be inside the working_dir.
    pdfgetter
        A PDFgetter to transform the chi file to gr file. If None, a default one will be created, see
        make_default_pdfgetter. Default None
    refine_func
        A function func(gr_file: str) to refine the G(r) of Ni. If None, a default refine function will be used, see
        refine. Default None

    Returns
    -------
    result_dct:
        A mapping from the wavelength to the result of the pipeline. It is also saved in the json file.

    Examples
    --------
    Make a folder "mycalib" as working directory. Make subdirectories in inside and in each subdirectory, copy a poni
    file into it. So you wil get a file system like the following:
        mycalib
            - subdir0
                - file0.poni
            - subdir1
                - file1.poni
            ...
    Then, get the path to the tiff file. Here, assuming the path to the tiff file is
    "tiff_base/mysample/darksub/mytiff.tiff", run the function by
    >>>  result = calib_wl("tiff_base/mysample/darksub/mytiff.tiff", "mycalib")
    The calibration will start. After it finishes, use "plot_rw_wl" to visualize the results.
    >>> plot_rw_wl(result)
    Or use the "summarize" to visual the result based on the json file "mycalib/result.json".
    >>> summarize("mycalib/result.json")
    """
    child_dirs = []
    for item in os.listdir(working_dir):
        item_path = os.path.join(working_dir, item)
        if os.path.isdir(item_path) and item[0] != '.':
            child_dirs.append(item_path)
    child_dirs = sorted(child_dirs)

    json_file = os.path.join(working_dir, json_file)
    if os.path.isfile(json_file):
        former_result = load_result(json_file)
        visited_dirs = [dct.get("directory") for dct in former_result.values()]
    else:
        visited_dirs = []

    final_result = {}
    res_gen = (
        run_pipe(tiff_file, child_dir, pdfgetter=pdfgetter, refine_func=refine_func)
        for child_dir in child_dirs
        if child_dir not in visited_dirs
    )
    for result in res_gen:
        final_result.update(result)
    dump_result(final_result, json_file)
    return result_dct


def run_pipe(tiff_file: str,
             saving_dir: str = ".",
             poni_file: str = None,
             pdfgetter: PDFGetter = None,
             refine_func: Callable = None) -> dict:
    """
    A pipeline process to integrate, transform and fit the data and return a dictionary of path to the resulting files.

    Parameters
    ----------
    tiff_file
        Path to the tiff file.
    saving_dir
        Path to the directory where the output data files will be saved. Default "."
    poni_file
        Path to the poni file. If None, the poni file in the saving_dir will be used.
    pdfgetter
        A PDFgetter to transform the chi file to gr file. If None, a default one will be created, see
        make_default_pdfgetter. Default None
    refine_func
        A function func(gr_file: str) to refine the G(r) of Ni. If None, a default refine function will be used, see
        refine. Default None

    Returns
    -------
    dct
        a dictionary that mapping the wavelength to a dict containing keys: "poni", "chi", "gr", "Rw", "csv", "fgr".

    Examples
    --------
    Move the poni file "example_poni.poni" into a directory "example".
    Get the path to a tiff file of the calibrant "example_tiff.tiff".
    >>> run_pipe("example_tiff.tiff", "example")
    The program will run the pipeline automatically according to the information in poin file.
    If user would like to change the settings inside the pipeline, for example,
    "fit_range" to a range from 0. A to 20. A with 0.01 A as a step.
    >>> REFINE_CONFIG["fit_range"] = (0., 20., 0.01)
    """
    def find_poni(dir_to_search):
        files_found = recfind(dir_to_search, r".*\.poni")
        if len(files_found) < 1:
            raise Exception(f"Not found poni file in {dir_to_search}")
        elif len(files_found) > 1:
            _poni_file = files_found[0]
            print(f"Multiple file found in {dir_to_search}.\nUse {_poni_file}")
        else:
            _poni_file = files_found[0]
        return _poni_file

    poni_file = poni_file if poni_file else find_poni(saving_dir)

    wl = load(poni_file)["Wavelength"] * 1e10  # unit A
    result = dict(poni=poni_file, directory=saving_dir)
    print(f"Run pipeline with wavelength: {wl:.4f} and poni file: {poni_file}")

    result["chi"], _ = xpdtools_int(poni_file, tiff_file, chi_dir=saving_dir, **XPDTOOLS_CONFIG)

    print("Transform the data ...")
    pdfgetter = pdfgetter if pdfgetter else make_default_pdfgetter()
    chi_q, chi_i = loaddata(result["chi"]).T
    pdfgetter(chi_q, chi_i)
    visualize(pdfgetter)
    filename = os.path.splitext(result["chi"])[0]
    for datatype in ("iq", "sq", "fq", "gr"):
        result[datatype] = path = filename + f".{datatype}"
        pdfgetter.writeOutput(path, datatype)

    if refine_func is None:
        refine_func = default_refine
    result["Rw"], result["csv"], result["fgr"] = refine_func(result["gr"], saving_dir)

    dct = {wl: result}
    return dct


def make_default_pdfgetter() -> PDFGetter:
    """
    Create a PDFgetter with default setting for Ni.
    """
    config = PDFConfig(**PDFGETTER_CONFIG)
    pdfgetter = PDFGetter(config)
    return pdfgetter


def default_refine(gr_file: str, res_dir: str) -> Tuple[float, str, str]:
    """
    Refine the G(r) using a default recipe for Ni.

    Parameters
    ----------
    gr_file
        Path to the gr file.
    res_dir
        Directory to save csv and fgr file.

    Returns
    -------
    rw
        Rw value.
    csv_file
        Path to fitting parameters csv file.
    fgr_file
        Path to fitted data fgr file.
    """
    file_name_base = os.path.splitext(os.path.basename(gr_file))[0]
    print(f"Refine {file_name_base}, please wait ...")

    default_ni_file = os.path.join(os.path.dirname(__file__), "data_files", "Ni.cif")
    ni = GenConfig(name="Ni",
                   stru_file=default_ni_file,
                   ncpu=4)
    config_ni = ConConfig(name="calibration",
                          data_id=0,
                          data_file=gr_file,
                          fit_range=REFINE_CONFIG["fit_range"],
                          eq=ni.name,
                          phases=ni,
                          qparams=REFINE_CONFIG["qparams"])
    recipe = make(config_ni)

    con: FitContribution = recipe.calibration
    gen: PDFGenerator = con.Ni

    recipe.addVar(gen.scale, value=REFINE_CONFIG["scale"])
    recipe.addVar(gen.delta2, value=REFINE_CONFIG["delta2"])
    recipe.addVar(gen.qdamp)
    recipe.addVar(gen.qbroad)
    sgpars = constrainAsSpaceGroup(gen.phase, REFINE_CONFIG["sgnum"])
    for par in sgpars.latpars:
        recipe.addVar(par, tag="lat")
    for par in sgpars.adppars:
        recipe.addVar(par, tag="adp", value=REFINE_CONFIG["uiso"])

    free_and_fit(recipe, ("scale", "lat"), "adp", "delta2", ("qdamp", "qbroad"), verbose=0)

    rw = float(FitResults(recipe).rw)
    base_name = os.path.join(res_dir, file_name_base)
    csv_file, fgr_file = old_save(recipe, "calibration", base_name)

    print(f"Result: Ni fitting (Rw = {rw:.3f})")
    plot(recipe)

    return rw, csv_file, fgr_file


def summarize(json_file: str):
    """
    Get the result data in json file and plot the Rw as a function of wavelength.

    Parameters
    ----------
    json_file
        The path to the json file of the results.
    """
    result = load_result(json_file)
    plot_rw_wl(result)
    return


def plot_rw_wl(result_dct: Dict[float, dict]):
    """
    Get data from result list and plot the Rw v.s wavelength.
    Parameters
    ----------
    result_dct
        result list of pipe line processing.
    """
    wls, rws = [], []
    for wl, result in sorted(result_dct.items(), key=lambda kv: kv[0]):
        wls.append(wl)
        rws.append(result["Rw"])

    plt.figure()
    plt.plot(wls, rws, "o-")

    plt.xlabel(r"wavelength ($\AA$)")
    plt.ylabel(r"Rw of Ni fitting")
    plt.show()
    return


def visualize(pdfgetter: PDFGetter):
    """
    Visualize the results from the pdfgetter.

    Parameters
    ----------
    pdfgetter
        The pdfgetter containing the data after data reduction.
    """
    from matplotlib.gridspec import GridSpec
    grids = GridSpec(2, 2)
    plt.figure(figsize=(8, 8))
    data_pairs = (
        pdfgetter.iq,
        pdfgetter.sq,
        pdfgetter.fq,
        pdfgetter.gr)
    xylabels = (
        (r"Q ($\AA^{-1}$)", r"I (A. U.)"),
        (r"Q ($\AA^{-1}$)", r"S"),
        (r"Q ($\AA^{-1}$)", r"F ($\AA^{-1}$)"),
        (r"r ($\AA$)", r"G ($\AA^{-2}$)")
    )
    for grid, data_pair, xylabel in zip(grids, data_pairs, xylabels):
        plt.subplot(grid)
        plt.plot(*data_pair)
        plt.xlabel(xylabel[0])
        plt.ylabel(xylabel[1])
    plt.show()
    return


def load_result(json_file):
    """Load the result from json file."""
    with open(json_file, "r") as f:
        res_lst = json.load(f)
    return res_lst


def dump_result(dct, json_file):
    """Dump the result to a json file."""
    with open(json_file, "w+") as f:
        json.dump(dct, f)
    return

