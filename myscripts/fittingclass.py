from typing import Union, List, Callable, Iterable, Tuple
from diffpy.srfit.fitbase import FitRecipe, ProfileGenerator


__all__ = ["GenConfig", "FunConfig", "ConConfig", "MyRecipe"]


class GenConfig:
    """
    A configuration class to provide information in the building of PDFGenerator or DebyePDFGenerator. It is used
    by 'make_generator' in 'myscripts.fittingfunction'.

    Attributes
    ----------
    name
        The name of the generator.
    stru_file
        The file path of the structure file.
    periodic : bool
        If the structure if periodic. Default if cif or stru, True else False.
    debye : bool
        Use DebyePDFGenerator or PDFGenerator. Default: if periodic, False else True
    ncpu : int
        number of parallel computing cores for the generator. If None, no parallel. Default None.

    """
    def __init__(self, name: str, stru_file: str, **kwargs):
        """
        Initiate the GenConfig.

        Parameters
        ----------
        name
            The name of the generator.
        stru_file
            The file path of the structure file.
        kwargs: (Optional) Keyword arguments to pass to the build_generator functions. They are
            periodic : bool
                If the structure if periodic. Default if cif or stru, True else False.
            debye : bool
                Use DebyePDFGenerator or PDFGenerator. Default: if periodic, False else True
            ncpu : int
                number of parallel computing cores for the generator. If None, no parallel. Default None.
        """
        self.name = name
        self.stru_file = stru_file
        self.check(kwargs)
        self.periodic = kwargs.get("periodic", self.is_periodic(stru_file))
        self.debye = kwargs.get("debye", not self.periodic)
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
        known_keywords = ["periodic", "debye", "ncpu"]
        for key in kwargs:
            if key in known_keywords:
                continue
            else:
                raise KeyError(f"Unknown keyword: {key}")
        return


class FunConfig:
    """
    Configuration for the characteristic function.

    Attributes
    ----------
        name
            name of the function, also the name in Fitcontribution.
        func_type
            characteristic function from diffpy cmi.
        argnames
            argument names in the function. it will rename all arguments to avoid conflicts. If None, no renaming.
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


class ConConfig:
    """
    Configuration for the FitContribution.

    Attributes
    ----------
        name
            the name of Fitcontribution.
        data_file
            path to the data file.
        fit_range
            rmin, rmax, rstep for fitting.
        qparams
            qdamp, qbroad from calibration.
        eq
            equation string for the Fitcontribution.
        phases
            single or a list of GenConfig object. Default empty tuple.
        functions
            single or a list of FunConfig object. Default empty tuple.
        base_lines
            single or a list of Generator instance of base line. Default empty tuple.
        res_eq
            string residual equation. Default "chiv".
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
        self.name: str = name
        self.eq: str = eq
        self.data_file: str = data_file
        self.fit_range: Tuple[float, float, float] = fit_range
        self.qparams: Tuple[float, float] = qparams
        self.phases: List[GenConfig] = _make_list(phases)
        self.functions: List[FunConfig] = _make_list(functions)
        self.base_lines: List[ProfileGenerator] = _make_list(base_lines)
        self.res_eq: str = res_eq


class MyRecipe(FitRecipe):
    """
    The FitRecipe with augmented features.

    Attributes
    ----------
    configs
        single or multiple configurations to initiate the contributions in recipe.
    res
        FitResult. Updated when FitResult is called. Default None.
    name
        name of the recipe. It will be saved if 'save' is used. Default None.
    """
    def __init__(self, configs: Tuple[ConConfig], name=None):
        """Initiate the class."""
        super().__init__()
        self.configs = configs
        self.res = None
        self.name = name


def _make_list(item) -> list:
    """
    if item is not a list or tuple, make it a list with only the item in it.
    :param item: object.
    :return:
    """
    if isinstance(item, (list, tuple)):
        pass
    else:
        item = [item]
    return item
