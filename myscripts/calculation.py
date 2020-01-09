from diffpy.srreal.scatteringfactortable import SFTXray
from diffpy.srreal.sfaverage import SFAverage
from diffpy.structure import loadStructure
from molmass import Formula
from typing import Union, List, Tuple, Dict, Iterable
from pandas import Series


__all__ = ["molar_fraction", "weight_ratio"]


XTB = SFTXray()


def calc_fs_from_stru_files(*stru_files: str, qa=0) -> List[float]:
    """Calculate the compositional average scattering factor from the structure file."""
    fs = []
    for stru_file in stru_files:
        stru = loadStructure(stru_file)
        f = SFAverage.fromStructure(stru, XTB, qa).f1avg
        fs.append(f)
    return fs


def calc_fs_from_comps(*comps: Formula, qa=0) -> List[float]:
    """Calculate the compositional average scattering factor from the compositions."""
    def to_dict(f: Formula):
        dct = {}
        c = f.composition()
        for tup in c:
            dct[tup[0]] = tup[1]
        return dct
    return [SFAverage.fromComposition(to_dict(comp), XTB, qa).f1avg for comp in comps]


def calc_molar_masses(*comps: Formula) -> List[float]:
    """Calculate the molar masses from the compositions."""
    return [comp.mass for comp in comps]


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
    f1avg = calc_fs_from_stru_files(*stru_files)
    sr = Series(f1avg, index=scale.index)
    ratio = scale / sr.pow(2)
    frac = ratio / ratio.sum()
    return frac


def weight_ratio(scale: Series, compositions: List[str]) -> Series:
    """
    Calculate the molar fractions of phases based on its scale factors and compositions or structures.

    Parameters
    ----------
    scale : Series
        A series of scale factors with the name of the phases as index.
    compositions : Iterable
        An Iterable of the compositions.

    Returns
    -------
    fraction : Series
        A series of weight ratios with the names of the phases as index.

    Examples
    --------
    >>> scale = pd.Series([0.3, 0.4], index=['scale_a', 'scale_b'])
    >>> composition = [{'atom_A': 1}, {'atom_B': 1}]
    >>> weight_ratio(scale, compositions)

    """
    formulas = [Formula(composition) for composition in compositions]
    f = calc_fs_from_comps(*formulas)
    f1avg = Series(f, index=scale.index)
    m = calc_molar_masses(*formulas)
    mass = Series(m, index=scale.index)

    ratio = scale / f1avg.pow(2)
    mol_frac = ratio / ratio.sum()
    ratio1 = mol_frac * mass
    fraction = ratio1 / ratio1.sum()
    return fraction
