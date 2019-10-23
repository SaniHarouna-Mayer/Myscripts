from typing import Union, List, Callable, Iterable, Tuple
from deprecated import deprecated
from diffpy.srfit.fitbase import FitRecipe
from pandas import DataFrame


@deprecated(version="1.0", reason="It is depecrated. Use GenConfig instead.")
class Phase:
    def __init__(self, name: str, stru_file: str, **kwargs):
        """
        initiate Phase object. Phase object is a data structure to store the information to build generators.
        :param name: name of the phase, also generator name in Fitrecipe.
        :param stru_file: file name of the structure file.
        :param kwargs: (Optional) keyword arguments to pass to the build_generator functions.
               periodic: (bool) if the structure if periodic. Default auto choose according to extension
               debye: (bool) use DebyePDFGenerator or PDFGenerator. Default auto choose according to extension
               qmin: (float) qmin for the generator. Default 0. for PDFGenerator and 1. for DebyePDFGenerator
               qmax: (float) qmax for the generator. Default read value from data file when making recipe
               ncpu: (int) number of parallel computing cores for the generator. If None, no parallel. Default None
        """
        self.name = name
        self.stru_file = stru_file
        self.check(kwargs)
        self.periodic = kwargs.get("periodic", self.is_periodic(stru_file))
        self.debye = kwargs.get("debye", not self.periodic)
        self.qmin = kwargs.get("qmin", None)
        self.qmax = kwargs.get("qmax", None)
        self.ncpu = kwargs.get("ncpu", None)

    @staticmethod
    def is_periodic(stru_file: str):
        """
        get periodicity from the extension of structure file.
        :param stru_file: structure file name.
        :return:
        """
        import os
        base = os.path.basename(stru_file)
        _, ext = os.path.splitext(base)
        if ext in (".xyz", ".mol"):
            decision = False
        elif ext in (".cif", ".stru"):
            decision = True
        else:
            decision = False
            message = "Unknown file type: {} ".format(ext) + \
                      "periodic = False"
            print(message)
        return decision

    @staticmethod
    def check(kwargs):
        """
        check if the keyword argument is known.
        :return:
        """
        known_keywords = ["periodic", "debye", "qmin", "qmax", "ncpu"]
        for key in kwargs:
            assert key in known_keywords, f"Unknown keyword: {key}"
        return


class GenConfig:
    def __init__(self, name: str, stru_file: str, **kwargs):
        """
        Generator configuration. It is used to make generator.
        :param name: name of the phase, also generator name in Fitrecipe.
        :param stru_file: file name of the structure file.
        :param kwargs: (Optional) keyword arguments to pass to the build_generator functions.
               periodic: (bool) if the structure if periodic. Default auto choose according to extension
               debye: (bool) use DebyePDFGenerator or PDFGenerator. Default auto choose according to extension
               qmin: (float) qmin for the generator. Default 0. for PDFGenerator and 1. for DebyePDFGenerator
               qmax: (float) qmax for the generator. Default read value from data file when making recipe
               ncpu: (int) number of parallel computing cores for the generator. If None, no parallel. Default None
        """
        self.name = name
        self.stru_file = stru_file
        self.check(kwargs)
        self.periodic = kwargs.get("periodic", self.is_periodic(stru_file))
        self.debye = kwargs.get("debye", not self.periodic)
        self.qmin = kwargs.get("qmin", None)
        self.qmax = kwargs.get("qmax", None)
        self.ncpu = kwargs.get("ncpu", None)

    @staticmethod
    def is_periodic(stru_file: str):
        """
        get periodicity from the extension of structure file.
        :param stru_file: structure file name.
        :return:
        """
        import os
        base = os.path.basename(stru_file)
        _, ext = os.path.splitext(base)
        if ext in (".xyz", ".mol"):
            decision = False
        elif ext in (".cif", ".stru"):
            decision = True
        else:
            decision = False
            message = "Unknown file type: {} ".format(ext) + \
                      "periodic = False"
            print(message)
        return decision

    @staticmethod
    def check(kwargs):
        """
        check if the keyword argument is known.
        :return:
        """
        known_keywords = ["periodic", "debye", "qmin", "qmax", "ncpu"]
        for key in kwargs:
            assert key in known_keywords, f"Unknown keyword: {key}"
        return


@deprecated(version="1.0", reason="It is deprecated. Use FunConfig instead.")
class Function:
    def __init__(self, name: str, func_type: Callable, argnames: List[str] = None):
        """
        initiate Function object. Function object is a data structure to store the information to register
        characteristic functions to contribution
        :param name: name of the function, also the name in Fitcontribution.
        :param func_type: characteristic function from diffpy cmi.
        :param argnames: (Optional) argument names in the function. it will rename all arguments to avoid conflicts.
        it always starts with "r" when using diffpy characteristic functions.
        """
        self.name = name
        self.func_type = func_type
        self.argnames = argnames


class FunConfig:
    """
    Configuration for the characteristic function.
    Attributes
        name: name of the function, also the name in Fitcontribution.
        func_type: characteristic function from diffpy cmi.
        argnames: argument names in the function. it will rename all arguments to avoid conflicts. If None, no renaming.
        If not None, it always starts with "r" when using diffpy characteristic functions. Default None.
    """
    def __init__(self, name: str, func_type: Callable, argnames: List[str] = None):
        """
        initiate Function object. Function object is a data structure to store the information to register
        characteristic functions to contribution
        :param name: attribute "name".
        :param func_type: attribute "func_type".
        :param argnames: (Optional) attribute "argnames".
        """
        self.name = name
        self.func_type = func_type
        self.argnames = argnames


@deprecated(version="1.0", reason="It is deprecated. Use ConConfig instead.")
class Model:
    def __init__(self,
                 name: str,
                 data_file: str,
                 fit_range: Tuple[float, float, float],
                 qparams: Tuple[float, float],
                 eq: str,
                 phases: Union[Phase, List[Phase]] = (),
                 functions: Union[Function, List[Function]] = (),
                 base_lines: Union[Callable, List[Callable]] = (),
                 res_eq: str = "chiv"):
        """
        initiate Model object, which is used to make the Fitcontribution.
        :param name: name for the model. it will be used as name of Fitcontribution.
        :param data_file: path to the data file.
        :param fit_range: rmin, rmax, rstep for fitting.
        :param qparams: qdamp, qbroad from calibration.
        :param eq: equation string for the Fitcontribution.
        :param phases: single or a list of Phase object. Default empty tuple.
        :param functions: single or a list of Function object. Default empty tuple.
        :param base_lines: single or a list of Generator of base line. Default empty tuple.
        :param res_eq: string residual equation. Default "chiv".
        """
        self.name = name
        self.eq = eq
        self.data_file = data_file
        self.fit_range = fit_range
        self.qparams = qparams
        self.phases = _make_list(phases)
        self.functions = _make_list(functions)
        self.base_lines = _make_list(base_lines)
        self.res_eq = res_eq


class ConConfig:
    """
    Configuration for the FitContribution.
    Attributes
        name: the name of Fitcontribution.
        data_file: path to the data file.
        fit_range: rmin, rmax, rstep for fitting.
        qparams: qdamp, qbroad from calibration.
        eq: equation string for the Fitcontribution.
        phases: single or a list of GenConfig object. Default empty tuple.
        functions: single or a list of FunConfig object. Default empty tuple.
        base_lines: single or a list of Generator instance of base line. Default empty tuple.
        res_eq: string residual equation. Default "chiv".
    """
    def __init__(self,
                 name: str,
                 data_file: str,
                 fit_range: Tuple[float, float, float],
                 qparams: Tuple[float, float],
                 eq: str,
                 phases: Union[GenConfig, List[GenConfig]] = (),
                 functions: Union[FunConfig, List[FunConfig]] = (),
                 base_lines: Union[Callable, List[Callable]] = (),
                 res_eq: str = "chiv"):
        """
        initiate Model object, which is used to make the Fitcontribution.
        :param name: name for the model. it will be used as name of Fitcontribution.
        :param data_file: path to the data file.
        :param fit_range: rmin, rmax, rstep for fitting.
        :param qparams: qdamp, qbroad from calibration.
        :param eq: equation string for the Fitcontribution.
        :param phases: single or a list of Phase object. Default empty tuple.
        :param functions: single or a list of Function object. Default empty tuple.
        :param base_lines: single or a list of Generator of base line. Default empty tuple.
        :param res_eq: string residual equation. Default "chiv".
        """
        self.name = name
        self.eq = eq
        self.data_file = data_file
        self.fit_range = fit_range
        self.qparams = qparams
        self.phases = _make_list(phases)
        self.functions = _make_list(functions)
        self.base_lines = _make_list(base_lines)
        self.res_eq = res_eq


class MyRecipe(FitRecipe):
    def __init__(self, *configs: ConConfig):
        """Initiate the class."""
        super().__init__()
        self.configs: Tuple[ConConfig] = configs
        self.res = None
        self.csv_df: DataFrame = DataFrame()
        self.fgr_df: DataFrame = DataFrame()
        self.cif_df: DataFrame = DataFrame()


def _make_list(item) -> list:
    """
    if item is not a list or tuple, make it a list with only the item in it.
    :param item: object.
    :return:
    """
    if isinstance(item, Iterable):
        pass
    else:
        item = [item]
    return item
