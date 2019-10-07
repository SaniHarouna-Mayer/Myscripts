from diffpy.pdfgetx import loaddata, PDFGetter, PDFConfig
from myint import xpdtools_int
from fittingfunction import *
from fittingclass import *
from yamlmaker import load
from helper import recfind


def calib_wl(tiff_file: str,
             working_dir: str,
             pdfgetter: PDFGetter = None,
             refine_func: Callable = None) -> List[dict]:
    """
    Calibrate wavelength by integrating data in tiff_file, transforming to G(r) and fit with Ni structure. Results will
    be plotted out as Rw v.s. wl.
    Parameters
    ----------
    tiff_file
        Path to the tiff file.
    working_dir
        Path to the working directory, which hosts multiple child directories, each of which contains a poni
        file for a wavelength.
    pdfgetter
        A PDFgetter to transform the chi file to gr file. If None, a default one will be created, see
        make_default_pdfgetter. Default None
    refine_func
        A function func(gr_file: str) to refine the G(r) of Ni. If None, a default refine function will be used, see
        refine. Default None

    Returns
    -------
        A list of dictionary that contains keys: "poni", "chi", "gr", "Rw", "csv", "fgr", "wl".
    """
    child_dirs = [os.path.join(working_dir, item_name)
                  for item_name in os.listdir(working_dir)
                  if os.path.isdir(item_name) and item_name[0] != r"."]
    child_dirs.sort()

    res_lst = list()

    for child_dir in child_dirs:
        result = run_pipe(tiff_file, child_dir, pdfgetter=pdfgetter, refine_func=refine_func)
        res_lst.append(result)
        print("-" * 100)

    return res_lst


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
    res_dct
        a dictionary that contains keys: "poni", "chi", "gr", "Rw", "csv", "fgr".
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

    if poni_file:
        pass
    else:
        poni_file = find_poni(saving_dir)

    wl = load(poni_file)["Wavelength"] * 1e10  # unit A
    result = dict(poni=poni_file, wl=wl)
    print(f"Run pipeline with wavelength: {wl:.4f} and poni file: {poni_file}")

    result["chi"], _ = xpdtools_int(poni_file, tiff_file, chi_dir=saving_dir)

    if pdfgetter:
        pass
    else:
        pdfgetter = make_default_pdfgetter()

    chi_q, chi_i = loaddata(result["chi"]).T
    pdfgetter(chi_q, chi_i)
    result["gr"] = os.path.splitext(result["chi"])[0] + ".gr"
    pdfgetter.writeOutput(result["gr"], "gr")

    if refine_func:
        pass
    else:
        refine_func = default_refine

    result["Rw"], result["csv"], result["fgr"] = refine_func(result["gr"], saving_dir)

    return result


def make_default_pdfgetter() -> PDFGetter:
    """
    Create a PDFgetter with default setting for Ni.
    """
    config = PDFConfig()
    config.composition = "Ni"
    config.dataformat = "QA"
    config.qmin = 0.
    config.qmax = 24.
    config.qmaxinst = 25.
    config.rmin = 0.
    config.rmax = 60.
    config.rstep = .01
    config.rpoly = 0.9

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
    print(f"Refine {file_name_base}...")

    ni = Phase("Ni", "Ni.cif")
    config_ni = Model(name="one_phase", data_file=gr_file, fit_range=(1., 60., .01), eq="Ni", phases=ni,
                      qparams=(0.04, 0.02))
    recipe = make(config_ni)

    con: FitContribution = recipe.one_phase
    gen: PDFGenerator = con.Ni

    recipe.addVar(gen.scale, value=0.4)
    recipe.addVar(gen.delta2, value=2.0)
    recipe.addVar(gen.qdamp)
    recipe.addVar(gen.qbroad)
    sgpars = constrainAsSpaceGroup(gen.phase, 225)
    for par in sgpars.latpars:
        recipe.addVar(par, tag="lat")
    for par in sgpars.adppars:
        recipe.addVar(par, tag="adp", value=0.006)

    recipe.fix("all")
    recipe.free("scale")
    fit(recipe, verbose=0)
    recipe.free("lat")
    fit(recipe, verbose=0)
    recipe.free("adp")
    fit(recipe, verbose=0)
    recipe.free("delta2")
    fit(recipe, verbose=0)
    recipe.free("qdamp", "qbroad")
    fit(recipe, verbose=0)

    rw = float(FitResults(recipe).rw)
    saving_path = os.path.join(res_dir, file_name_base)
    csv_file, fgr_file = save(recipe, "one_phase", saving_path)

    plot(recipe.one_phase)
    plt.title(f"Ni fitting (Rw = {rw:.3f})")
    plt.show()

    return rw, csv_file, fgr_file


def plot_rw_wl(res_lst: List[dict]) -> None:
    """
    Get data from result list and plot the Rw v.s wavelength.
    Parameters
    ----------
    res_lst
        result list of pipe line processing.

    Returns
    -------
    None.
    """
    wls = [res["wl"] for res in res_lst]
    rws = [res["Rw"] for res in res_lst]

    plt.figure()
    plt.plot(wls, rws, "o-")

    plt.xlabel(r"wavelength ($\AA$)")
    plt.ylabel(r"Rw of Ni fitting")

    plt.show()
    return
