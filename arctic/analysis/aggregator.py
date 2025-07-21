from typing import Union, Callable

import pandas as pd


def apply_aggregation(df: pd.DataFrame, agg_func: Union[str, Callable]) -> Union[pd.DataFrame, pd.Series]:
    r"""
    Applies an aggregation function to a DataFrame.

    :param df: The DataFrame to aggregate
    :type df: pd.DataFrame
    :param agg_func: Aggregation function to apply, can be a string ('mean', 'median', 'sum') or a callable
    :type agg_func: str or callable

    :raises ValueError: If agg_func is not a valid aggregation function

    :return: The aggregated DataFrame or Series
    :rtype: pd.DataFrame or pd.Series
    """
    if callable(agg_func):
        return df.apply(agg_func)
    elif isinstance(agg_func, str) and hasattr(df, agg_func):
        return getattr(df, agg_func)()
    else:
        raise ValueError("Invalid 'agg_func'. Use 'mean', 'median', 'sum', or a callable function.")
