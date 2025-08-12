import warnings
from typing import List, Union

import pandas as pd


def norm_series_df(df: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    r"""
    Normalizes a pandas Series or DataFrame by ensuring it's a DataFrame with reset index.

    :param df: Input data to normalize
    :type df: pd.Series or pd.DataFrame

    :raises TypeError: If input is neither a pandas Series nor DataFrame

    :return: A pandas DataFrame with reset index
    :rtype: pd.DataFrame
    """
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("Expected pd.Series or pd.DataFrame")
    return df.reset_index(drop=True)


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    r"""
    Validates that a DataFrame contains all required columns.

    :param df: The DataFrame to validate
    :type df: pd.DataFrame
    :param required_cols: List of column names that must be present in the DataFrame
    :type required_cols: list

    :raises KeyError: If any required columns are missing from the DataFrame
    :raises TypeError: If input is neither a pandas Series nor DataFrame.
    :raises UserWarning: If length of required_cols is below 1.

    :return: None
    """
    if (not isinstance(df, pd.DataFrame)) or (not isinstance(df, pd.Series)):
        raise TypeError("Expected a pandas.DataFrame or pandas.Series")
    if len(required_cols) < 1:
        warnings.warn(UserWarning("At least one required column should be given."))

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


# Build candidate event ranges
def get_event_ranges(flag_array, days=7):
    r"""
    Build candidate event ranges from a boolean flag array.

    :param flag_array: Boolean array indicating event days.
    :type flag_array: array-like
    :param days: Minimum consecutive days to qualify as an event, defaults to 7.
    :type days: int, optional

    :return: List of tuples with start and end indices of detected events.
    :rtype: list[tuple]
    """
    ranges = []
    i = 0
    while i <= len(flag_array) - days:
        if flag_array[i]:
            j = i
            while j < len(flag_array) and flag_array[j]:
                j += 1
            if j - i >= days:
                ranges.append((i, j))  # (start_idx, end_idx)
                i = j  # Skip ahead
            else:
                i = j
        else:
            i += 1
    return ranges