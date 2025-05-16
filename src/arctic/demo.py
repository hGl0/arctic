import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

def read_data(input_file: str, enc: str = "utf-8") -> pd.DataFrame:
    """
    Reads data from a file, filtering only lines that start with 'D'.

    This function skips all lines starting with characters other than 'D'
    and converts the filtered lines into a pandas DataFrame.

    :param input_file: Path to the input file to read
    :type input_file: str
    :param enc: Encoding of the input file, defaults to "utf-8"
    :type enc: str, optional

    :return: DataFrame containing the filtered data
    :rtype: pd.DataFrame
    """
    with open(input_file, "r", encoding=enc) as f:
        filtered_lines = [line for line in f if line.startswith("D")]

    # Convert filtered lines into DataFrame
    from io import StringIO
    df = pd.read_csv(
        StringIO("".join(filtered_lines)),
        delimiter=",",
        low_memory=False
    )
    return df


def no_white_space(df: pd.DataFrame, sep: str = ' ') -> None:
    """
    Removes leading whitespace or specified separator from DataFrame column names.

    This function splits each column name by the separator and takes the last part,
    effectively removing any leading parts separated by the separator.

    :param df: DataFrame whose column names will be modified
    :type df: pd.DataFrame
    :param sep: Separator to split column names by, defaults to space
    :type sep: str, optional

    :return: None (modifies df in-place)
    """
    col = []
    for c in df.columns:
        col.append(c.split(sep)[-1])
    df.columns = col


def to_date(df: pd.DataFrame, col: str, format: str = 'mixed') -> None:
    """
    Converts string date values in a DataFrame column to datetime objects.

    :param df: DataFrame containing the column to convert
    :type df: pd.DataFrame
    :param col: Name of the column to convert to datetime
    :type col: str
    :param format: Date format string or 'mixed' for automatic parsing, defaults to 'mixed'
    :type format: str, optional

    :return: None (modifies df in-place)
    """
    df[col] = pd.to_datetime(df[col], format=format)
