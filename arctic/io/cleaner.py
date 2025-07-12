import pandas as pd

def no_white_space(df: pd.DataFrame, sep: str = ' ') -> pd.DataFrame:
    """
    Removes leading whitespace or specified separator from DataFrame column names.

    This function splits each column name by the separator and takes the last part,
    removing any leading parts separated by the separator.

    :param df: DataFrame whose column names will be modified
    :type df: pd.DataFrame
    :param sep: Separator to split column names by, defaults to space
    :type sep: str, optional

    :return: None (modifies df in-place)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a DataFrame, but {type(df)} was given.")

    col = []
    for c in df.columns:
        col.append(c.split(sep)[-1])
    df.columns = col

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

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
