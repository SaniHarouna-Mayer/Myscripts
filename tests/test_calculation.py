import myscripts.calculation as calculation
import pandas as pd


def test_weight_ratio():
    scale = pd.Series([1., 1., 1.])
    compositions = ['Al(OH)2', 'Al(OH)2', 'Al(OH)2']
    weight_ratio = calculation.weight_ratio(scale, compositions)
    assert isinstance(weight_ratio, pd.Series)
    assert weight_ratio.to_list() == [1./3, 1./3, 1./3]


if __name__ == "__main__":
    test_weight_ratio()