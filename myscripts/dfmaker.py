import os
import fire
import pandas as pd
from myscripts.tools import find_all_tiff


__all__ = ["make_df", "main"]


def make_df(tiff_base: str, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Make a DataFrame of tiff file paths and their corresponded poni file and tiff background. The DataFrame contains
    column: 'tiff' (path to tiff files, usually in a hard disk), 'poni' (path to the poni file used to integrate the
    tiff), 'bg' (path to the tiff background files for the background subtraction).

    Parameters
    ----------
    tiff_base
        Path to the tiff_base folder.
    mapping
        A standard DataFrame to map the directory name to the poni file path and tiff background file path.
        The DataFrame contains column: folder (name of the sample folders), 'poni' (poni file paths), 'bg' (background
        tiff file path)

    Returns
    -------
    df:
        a DataFrame of tiff file paths and their corresponded poni file and tiff background.
    """
    lst = []
    for _, row in mapping.iterrows():
        data_path = os.path.join(tiff_base, row['folder'])
        for tiff_path in find_all_tiff(data_path):
            dct = {'sample': row['folder'], 'bg': row['bg'], 'poni': row['poni'], 'tiff': tiff_path}
            lst.append(dct)
    df = pd.DataFrame(lst)
    return df.reindex(['sample', 'tiff', 'poni', 'bg'], axis=1)


def main(tiff_base: str, in_csv: str, out_csv: str):
    """
    Use make_df to make DataFrame of tiff files and their corresponding information according to in_csv and save in
    out_csv.

    Parameters
    ----------
    tiff_base
        Path to the tiff_base folder.
    in_csv
        Path to the input csv file. It includes columns: 'folder', 'bg', 'poni'.
    out_csv
        Path to the output csv file.

    Returns
    -------
        None.
    """
    mapping = pd.read_csv(in_csv)
    df = make_df(tiff_base, mapping)
    df.to_csv(out_csv, index=False)
    return


if __name__ == "__main__":
    fire.Fire(main)
