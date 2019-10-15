from diffpy.srfit.fitbase import ProfileGenerator
import numpy as np
from typing import Union, List, Callable, Iterable, Tuple


class ExpcosGenerator(ProfileGenerator):
    """
    baseline generator. simulate the baseline with a wave function:
    Gaussian(center, *std) * Sinwave(wavelength, cos_amp, sin_amp)
    *std means that the left right parts have different standard deviation.
    The attribute is the parameter in equation named with
    """
    def __init__(self, name):
        ProfileGenerator.__init__(self, name)
        self._newParameter("wavelength", 10.0)
        self._newParameter("center", 2.0)
        self._newParameter("cos_coef", .1)
        self._newParameter("sin_coef", .1)
        self._newParameter("left_std", 5.)
        self._newParameter("right_std", 5.0)
        return

    def __call__(self, x):
        a0 = self.wavelength.value
        a1 = self.center.value
        a2 = self.cos_coef.value
        a3 = self.sin_coef.value
        a4 = self.left_std.value
        a5 = self.right_std.value
        msk = x < a1
        x0 = x[msk]
        x1 = x[np.logical_not(msk)]
        env = np.concatenate([np.exp(-.5*((x0-a1)/a4)**2), np.exp(-.5*((x1-a1)/a5)**2)])
        osc = a2 * np.cos(2.*np.pi*x/a0) + a3 * np.sin(2.*np.pi*x/a0)
        y = env * osc
        return y


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
        self.phases = make_list(phases)
        self.functions = make_list(functions)
        self.base_lines = make_list(base_lines)
        self.res_eq = res_eq


def make_list(item) -> List:
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
