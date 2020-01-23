import os
import pandas as pd
import pyperclip
from typing import List, Iterable, Union, Callable
import matplotlib.pyplot as plt


__all__ = ['find_all_tiff', 'find_all', 'to_report', 'summarize', 'get_result']


def find_all_tiff(dir_path: str) -> Iterable[str]:
    """
    Yield the paths of all non-hidden tiff files in a directory.

    Parameters
    ----------
    dir_path
        path to the directory.

    Yields
    -------
    tiff_path:
        path to tiff files.

    """
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(".tiff") and not filename.startswith("."):
                filepath = os.path.join(root, filename)
                yield filepath


def find_all(dir_path: str, ext: str) -> Iterable[str]:
    """
    Yield the paths of all non-hidden files with certain extension in a directory.

    Parameters
    ----------
    dir_path
        path to the directory.
    ext
        extension.

    Yields
    -------
    tiff_path:
        path to files.

    """
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            if filename.endswith(f".{ext}") and not filename.startswith("."):
                filepath = os.path.join(root, filename)
                yield filepath


def summarize(csv_df: pd.DataFrame, phases_col: str = "phases", file_col: str = 'csv_file', data_col: str = 'val',
              name_col: str = None, exclude: List[str] = None, map_phase: dict = None,
              phase_del: str = ', ', escape: bool = False, processing: Callable[[pd.DataFrame], None] = None,
              printout: bool = False, cbcopy: bool = False, output: bool = False) -> Union[None, pd.DataFrame]:
    """
    Group the results by phases. For each group data frame, read .csv files according to the csv_df. Each .csv file
    contains fitting results from a fit. Join the results into one large data frame. The index are the parameters and
    the columns are the fits. Print the data frame out to use in the report. It can also copy to clip board or return
    a list of data frame.

    Parameters
    ----------
    csv_df
        The data frame of the csv files and its meta data, including phases and other information.
    file_col
        The column name of the .csv file.
    data_col
        The column name of values of the fitting parameters.
    name_col
        The column name for the names of columns in the result data frame.
    phases_col
        The a string of phases used in fitting.
    exclude
        The phases not to show results. If None, all phases will be included. Default None.
    map_phase
        The mapping from the phase name to a new name. If None, the phase name will be striped of '_' and capitalized.
         Default None.
    phase_del
        The delimiter of the phase in the string of phases. Dafault ', '.
    escape
        By default, the value will be read from the pandas config module. When set to False prevents from escaping
        latex special characters in column names. Default False.
    processing
        A function to modify the result data frame for each phase in each sample. If None, no modification.
        Default None.
    printout
        If True, printout the result string. Default False.
    cbcopy
        If True, copy result string to the clip board. Default False.
    output
        If True, return a list of the results data_frames. Default False.

    Returns
    -------
    None or list of data frame.

    """
    if not printout and not cbcopy and not output:
        raise Warning("printout, cbcopy and output are all None. Do nothing.")

    def gen_latex_str():
        for phases_str, group in csv_df.groupby(phases_col):
            _res_df = get_result(group, file_col=file_col, data_col=data_col, name_col=name_col)
            _res_df.drop(index=['Rw', 'half_chi2'], inplace=True)
            latex_str = to_report(_res_df, phases_str=phases_str, exclude=exclude, map_phase=map_phase,
                                  phase_del=phase_del, escape=escape, processing=processing)
            yield _res_df, latex_str

    res_dfs = []
    all_lstr = ''
    for res_df, lstr in gen_latex_str():
        res_dfs.append(res_df)
        all_lstr += lstr

    if printout:
        print(all_lstr)
    if cbcopy:
        pyperclip.copy(all_lstr)

    return res_dfs if output else None


def get_result(csv_df: pd.DataFrame, file_col: str = 'csv_file', data_col: str = 'val', name_col: str = None)\
        -> pd.DataFrame:
    """
    Get .csv files information from csv_df. Read multiple csv files. Each .csv file contains fitting results from a fit.
    Join the results into one large data frame. The index are the parameters and the columns are the fits.

    Parameters
    ----------
    csv_df
        The data frame of the csv files and its meta data, including phases and other information.
    file_col
        The column name of the .csv file.
    data_col
        The column name of values of the fitting parameters.
    name_col
        If specified, assign the names of columns in the result data frame.

    Returns
    -------
    res_df
        The results data frame.

    """
    dfs = (pd.read_csv(f, index_col=0)[data_col] for f in csv_df[file_col])
    res_df: pd.DataFrame = pd.concat(dfs, axis=1, ignore_index=True, sort=False).rename_axis(None)
    if name_col is not None:
        mapping = dict(zip(res_df.columns, csv_df[name_col]))
        res_df.rename(columns=mapping, inplace=True)

    return res_df


def to_report(res_df: pd.DataFrame, phases_str: str, exclude: List[str] = None, map_phase: dict = None,
              phase_del: str = ', ', escape: bool = False, processing: Callable[[pd.DataFrame], pd.DataFrame] = None)\
        -> str:
    """
    Split the data frame of results by the phases and make a dictionary mapping from phase to data frame of results.

    Parameters
    ----------
    res_df
        The data frame of Results.
    phases_str
        The a string of phases used in fitting.
    exclude
        The phases not to show results. If None, all phases will be included. Default None.
    map_phase
        The mapping from the phase name to a new name. If None, the phase name will be striped of '_' and capitalized.
         Default None.
    phase_del
        The delimiter of the phase in the string of phases. Dafault ', '.
    escape
        By default, the value will be read from the pandas config module. When set to False prevents from escaping
        latex special characters in column names. Default False.
    processing


    Returns
    -------
    latex_str
        A string of latex table.

    """
    def default_change(s):
        s = ' '.join(s.split('_'))
        s = s.capitalize()
        return s

    total_lines = []
    map_phase = {} if map_phase is None else map_phase
    exclude = {} if exclude is None else exclude
    phases = [phase for phase in phases_str.split(phase_del) if phase not in exclude]
    for n, phase in enumerate(phases):
        df = res_df.filter(like=phase, axis=0).copy()
        df.rename(index=convert_index, inplace=True)
        if processing is not None:
            df = processing(df)
        phase = map_phase.get(phase, default_change(phase))
        if n == 0:
            lines = to_lines(df, del_head=False, escape=escape)
            lines = lines[:4] + [phase + r'\\', r'\hline'] + lines[4:]
        elif n == len(phases) - 1:
            lines = to_lines(df, del_tail=False, escape=escape)
            lines = [r'\hline', phase + r'\\', r'\hline'] + lines
        else:
            lines = to_lines(df, escape=escape)
            lines = [r'\hline', phase + r'\\', r'\hline'] + lines
        total_lines += lines
    tabular_str = '\n'.join(total_lines)
    table_str = convert_str(tabular_str)
    return table_str


def convert_str(tabular_str: str) -> str:
    """
    Convert the tabular string to the table by adding the head and tail. Change the 'rule' to 'hline'.

    Parameters
    ----------
    tabular_str
        The string of tabular.

    Returns
    -------
    table_str
        The string for latex table.

    """
    str_map = {r"\toprule": r"\hline\hline",
               r"\midrule": r"\hline",
               r"\bottomrule": r"\hline\hline"}
    for old, new in str_map.items():
        tabular_str = tabular_str.replace(old, new)

    table_str = "\\begin{table}[htb]\n" + \
                "\\caption{}\n" + \
                "\\label{tab:}\n" + \
                tabular_str + \
                "\\end{table}"

    return table_str


def convert_index(index: str) -> str:
    """
    Map an index value to the report format.

    Parameters
    ----------
    index
        The name of the index.

    Returns
    -------
    new_index
        The new index in report format

    """
    usymbols = ['Uiso', 'U11', 'U12', 'U13', 'U21', 'U22', 'U23', 'U31', 'U32', 'U33']
    xyzsymbols = ['x', 'y', 'z']
    dsymbols = ['delta2', 'delta1']
    words = index.split('_')  # the last one is phase, don't need it
    if words[0] in ('a', 'b', 'c'):
        new_index = r'{} (\AA)'.format(words[0])
    elif words[0] in ('alpha', 'beta', 'gamma'):
        new_index = r'$\{}$ (deg)'.format(words[0])
    elif words[0] in usymbols:
        new_index = r'{}$_{{{}}}$({}) (\AA$^2$)'.format(words[0][0], words[0][1:], words[1])
    elif words[0] in xyzsymbols:
        new_index = r'{}$_{{{}}}$ (\AA)'.format(words[0], words[1])
    elif words[0] in dsymbols:
        new_index = r'$\{}_{}$ (\AA$^2$)'.format(words[0][:-1], words[0][-1])
    elif words[0] == 'psize':
        new_index = r'D (\AA)'
    else:
        new_index = words[0]
    return new_index


def to_lines(df: pd.DataFrame, del_head=True, del_tail=True, escape=False) -> List[str]:
    """
    Convert the results data frame to a list of lines in the string of latex table.

    Parameters
    ----------
    df
        The data frame of the results. Index are the parameters and columns are the samples.
    del_head
        Whether to delete the first four lines in latex string. Default True.
    del_tail
        Whether to delete the last two lines in latex string. Default True.
    escape
        By default, the value will be read from the pandas config module. When set to False prevents from escaping
        latex special characters in column names. Default False.

    Returns
    -------
    lines
        A list of lines in the latex table.

    """
    origin_str = df.to_latex(escape=escape)
    lines = origin_str.split('\n')
    if del_head:
        lines = lines[4:]
    if del_tail:
        lines = lines[:-3]
    return lines


def set_savefig(**settings):
    """
    Set kwargs for savefig and return a function to save figures..

    Parameters
    ----------
    settings
        settings that can be passed to 'func'. Usually:
            fig
                figure to save. If None, get current figure.
            savedir
                directory to save the figure
            fmt
                format of the figure. It will also be the extension.
            print_tex
                If True, choose to print latex or not. Default True.
            cbcopy
                If True, copy to clip board. Default True.

    Returns
    -------
    _savefig
        a function to save figures.

    """
    def _savefig(filename, *args, **kwargs):
        __doc__ = savefig.__doc__
        return savefig(filename, *args, **kwargs, **settings)
    return _savefig


def savefig(filename, caption="", fig=None, savedir=".", fmt="pdf", print_tex=True, cbcopy=True) -> None:
    """
    Save current figure and print out the latex text of loading figure or copy it to clip board.

    Parameters
    ----------
    filename
        name of the file without extension.
    caption
        caption to print in latex.
    fig
        figure to save. If None, get current figure.
    savedir
        directory to save the figure
    fmt
        format of the figure. It will also be the extension.
    print_tex
        If True, choose to print latex or not. Default True.
    cbcopy
        If True, copy to clip board. Default True.

    Returns
    -------
        None.

    """
    fig = fig if fig else plt.gcf()

    filename_with_ext = f"{filename}.{fmt}"
    filepath = os.path.join(savedir, filename_with_ext)
    fig.savefig(filepath, format=fmt)

    caption = "{" + caption + "}"
    label = "{" + "fig:" + filename + "}"
    file_in_tex = "{" + filename_with_ext + "}"
    tex = "\\begin{figure}[htb]\n" + \
          f"\\includegraphics[width=\\columnwidth]{file_in_tex}\n" + \
          f"\\caption{caption}\n" + \
          f"\\label{label}\n" + \
          "\\end{figure}"

    if print_tex:
        print(tex)
    if cbcopy:
        pyperclip.copy(tex)
    return
