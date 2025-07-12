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

    :raises ValueError: If any required columns are missing from the DataFrame

    :return: None
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pd.DataFrame")
    if len(required_cols) < 1:
        warnings.warn(UserWarning("At least one required column should be given."))

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
