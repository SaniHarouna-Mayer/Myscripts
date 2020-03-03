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
    stru_type
        The type of the structure object for the structure data to be loaded into. Choose from "diffpy" (
        DiffpyStructure), "crystal" (ObjCrystCrystal), "molecule" (ObjCrystMolecule).
    periodic : bool
        If the structure if periodic. Default if cif or stru, True else False.
    debye : bool
        Use DebyePDFGenerator or PDFGenerator. Default: if periodic, False else True
    sg : Union[int, str]
        The spacegroup of the structure. If int, it is the space group number. If string, it is the name in H-M.
    ncpu : int
        number of parallel computing cores for the generator. If None, no parallel. Default None.

    """
    def __init__(self, name: str, stru_file: str, stru_type: str = "diffpy", **kwargs):
        """
        Initiate the GenConfig.

        Parameters
        ----------
        name
            The name of the generator.
        stru_file
            The file path of the structure file.
        stru_type
            The type of the structure object for the structure data to be loaded into. Choose from "diffpy" (
            DiffpyStructure), "crystal" (ObjCrystCrystal), "molecule" (ObjCrystMolecule).
        kwargs: (Optional) Keyword arguments to pass to the build_generator functions. They are
            periodic : bool
                If the structure if periodic. Default if cif or stru, True else False.
            debye : bool
                Use DebyePDFGenerator or PDFGenerator. Default: if periodic, False else True
            sg : int, str
                The spacegroup of the structure. Either spacegroup number or name in H-M. Default: read spacegroup
                number and then spacegroup name from the stru_file. If not found, use 1, which is 'P 1' symmetry.
            ncpu : int
                number of parallel computing cores for the generator. If None, no parallel. Default None.
        """
        self.check(kwargs)
        self.name = name
        self.stru_file = stru_file
        self.stru_type = stru_type
        self.periodic = kwargs.get("periodic", self.is_periodic(stru_file))
        self.debye = kwargs.get("debye", not self.periodic)
        self.sg = kwargs.get("sg", self.read_sg(stru_file))
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

    def to_dict(self) -> dict:
        """
        Parse the GenConfig as a dictionary. The keys include: gen_name, stru_file, debye, periodic, ncpu.

        Returns
        -------
        config_dct
            A dictionary of generator configuration.

        """
        config_dct = {
            'gen_name': self.name,
            'stru_file': self.stru_file,
            'stru_type': self.stru_type,
            'debye': self.debye,
            'periodic': self.periodic,
            'ncpu': self.ncpu,
        }
        return config_dct

    @staticmethod
    def read_sg(stru_file: str) -> Union[int, str]:
        """
        Read the space group from the structure file. If not found, return 'P 1'
        1.

        Parameters
        ----------
        stru_file : str
            The path to the structure file.

        Returns
        -------
        sg : int, str
        """
        sg = 'P 1'
        with open(stru_file, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if '_symmetry_space_group_name_h-m' in line.lower():
                    sg = line.split()[1].strip('\'\"')  # second word without quotes
        return sg


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
        data_id
            The id of the data. It will be used as a foreign key when the results are saved.
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
                 data_id: int,
                 data_file: str,
                 fit_range: Tuple[float, float, float],
                 qparams: Tuple[float, float],
                 eq: str,
                 phases: Union[GenConfig, List[GenConfig]] = (),
                 functions: Union[FunConfig, List[FunConfig]] = (),
                 base_lines: Union[Callable, List[Callable]] = (),
                 res_eq: str = "chiv"):
        """
        Initiate the instance.

        Parameters
        ----------
        name
            the name of Fitcontribution.
        data_id
            The id of the data. It will be used as a foreign key when the results are saved.
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
        self.name: str = name
        self.data_id: int = data_id
        self.data_file: str = data_file
        self.fit_range: Tuple[float, float, float] = fit_range
        self.qparams: Tuple[float, float] = qparams
        self.eq: str = eq
        self.phases: List[GenConfig] = _make_list(phases)
        self.functions: List[FunConfig] = _make_list(functions)
        self.base_lines: List[ProfileGenerator] = _make_list(base_lines)
        self.res_eq: str = res_eq

    def to_dict(self) -> dict:
        """
        Parse the ConConfig to a dictionary. The keys will include: con_name, data_id, data_file, rmin, rmax, dr, qdamp,
        qbroad, eq, phases, functions, base_lines, res_eq.

        Returns
        -------
        config_dct
            A dictionary of configuration of FitContribution.
        """
        config_dct = {
            'con_name': self.name,
            'data_id': self.data_id,
            'data_file': self.data_file,
            'rmin': self.fit_range[0],
            'rmax': self.fit_range[1],
            'dr': self.fit_range[2],
            'qdamp': self.qparams[0],
            'qbroad': self.qparams[1],
            'eq': self.eq,
            'phases': ', '.join([gen.name for gen in self.phases]),
            'functions': ', '.join([fun.name for fun in self.functions]),
            'baselines': ', '.join([bl.name for bl in self.base_lines]),
            'res_eq': self.res_eq
        }
        return config_dct


class MyRecipe(FitRecipe):
    """
    The FitRecipe with augmented features.

    Attributes
    ----------
    configs
        single or multiple configurations to initiate the contributions in recipe.
    """
    def __init__(self, configs: Tuple[ConConfig]):
        """Initiate the class."""
        super().__init__()
        self.configs = configs


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
