import os
import pandas as pd
from typing import Set


def quote_str(string: str) -> str:
    return f'"{string}"'


def set_str(input: Set[str]) -> str:
    if len(input) == 0:
        return "{}"
    else:
        return str(input)


def dump_df(fname, df, index=True):
    """
    Writes a pandas DataFrame to a comma-separated values (.csv) file.

    :param fname: a string file name to write to
    :param df: a pandas DataFrame to write to file
    :param index: a bool functioning exactly as in pandas.DataFrame.to_csv()
    """
    os.makedirs(os.path.split(fname)[0], exist_ok=True)
    df.to_csv(fname, sep=',', index=index)


def load_df(fname, index_col=0):
    """
    Reads a pandas DataFrame from a comma-separated values (.csv) file.

    :param fname: a string file name to read from
    :param index_col: functions exactly as in pandas.read_csv()
    :returns: a pandas DataFrame
    """
    return pd.read_csv(fname, sep=',', index_col=index_col)
