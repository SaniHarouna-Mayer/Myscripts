from diffpy.srreal.scatteringfactortable import SFTXray
from diffpy.srreal.sfaverage import SFAverage
from diffpy.structure import loadStructure, Structure
from typing import Union, List, Tuple, Dict, Iterable, Callable
from pandas import Series
from uncertainties import ufloat


__all__ = ["sfaverage", "molar_fraction"]


def apply_to_multi(func: Callable) -> Callable[[Iterable], List]:
    """A decorator to make the function 'func(obj)' applicable to multiple objects (like a list or numpy array)"""
    def _func(objs: Iterable) -> List:
        lst = []
        for obj in objs:
            lst.append(func(obj))
        return lst
    return _func


def sfaverage(composition: Union[List[Tuple[str, float]], Dict[str, float]] = None,
              structure: Structure = None, qa=0) -> SFAverage:
    """Instantiate a x-ray SFAverage from the composition or structure."""
    xtb = SFTXray()
    if composition and structure:
        raise ValueError("Only one of the composition and structure can be used.")
    elif composition:
        sfavg = SFAverage.fromComposition(composition, xtb, qa)
    elif structure:

        sfavg = SFAverage.fromStructure(structure, xtb, qa)
    else:
        raise ValueError("Both composition and structure are None")
    return sfavg


@apply_to_multi
def calc_f1avg(stru_file: str = None) -> float:
    """Calculate the compositional average scattering factor from the structure file."""
    structure = loadStructure(stru_file)
    return sfaverage(structure=structure).f1avg


def molar_fraction(scale: Series, stru_files: Iterable[str] = None) -> Series:
    """
    Calculate the molar fractions of phases based on its scale factors and compositions or structures.

    Parameters
    ----------
    scale : Series
        A series of scale factors with the name of the phases as index.
    stru_files : Iterable
        An Iterable of the paths to the structure file of the phases in 'scale'.

    Returns
    -------
    frac : Series
        A series of molar fractions with the names of the phases as index.

    Examples
    --------
    >>> scale = pd.Series([0.3, 0.4], index=['scale_a', 'scale_b'])
    >>> stru_files = ['a.cif', 'b.cif']
    >>> molar_fraction(scale, stru_files=stru_files)

    """
    f1avg: Series = Series(calc_f1avg(stru_files), index=scale.index)
    ratio = scale / (f1avg ** 2)
    frac = ratio / ratio.sum()
    return frac
