import pandas as pd
from pandas.io.formats.style import Styler
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import warnings

from arctic.io.paths import check_path


def plot_correlation(df: pd.DataFrame,
                     **kwargs) -> Optional[Styler]:
    r"""
    Plots a correlation matrix for a given DataFrame.
    Recommended to use equally perceived colormaps with 0 as white, according to https://matplotlib.org/stable/users/explain/colors/colormaps.html
    :param df: Input pandas DataFrame
    :param kwargs:
                - savefig (str, optional): File path to save the correlation matrix as picture. Default is None.
                - cmap (str, optional): Colormap for visualisation. Default is 'RdBu'.
    :raises TypeError: If the input DataFrame is not a pandas DataFrame, or 'savefig' is not a string.
    :raises FileNotFoundError: If 'savefig' is not a valid path.
    :raises AttributeError: If the input DataFrame is not a pandas DataFrame.
    :return: pd.io.formats.style.Styler: Styled correlation matrix for Jupyter display, or None if an error occurs.
    """

    savefig = kwargs.get('savefig', None)
    savecsv = kwargs.get('savecsv', None)
    cmap = kwargs.get('cmap', 'RdBu')

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected input 'df' to be a Pandas DataFrame")

    if df.empty:
        warnings.warn("The provided DataFrame is empty. Returning None.")
        return None

    if not isinstance(cmap, str) or cmap not in plt.colormaps():
        warnings.warn(f"Invalid colormap '{cmap}'. Defaulting to 'RdBu'.")
        cmap = 'RdBu'

    try:
        correlation_matrix = df.corr(numeric_only=True)
        try:
            styled_matrix = correlation_matrix.style.background_gradient(cmap=cmap, vmin=-1, vmax=1.0)
        except AttributeError as e:
            raise AttributeError(f"Attribute Error {e}: Expected 'df' to be a Pandas DataFrame")
    except Exception as e:
        raise Exception(f"An error occurred while generating the correlation matrix: {e}")

    if savefig:
        # validate save_path
        check_path(savefig)

        fig, ax = plt.subplots(figsize=(15, 15))

        ax.matshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1.0)

        ticks = np.arange(len(correlation_matrix.columns))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(correlation_matrix.columns, rotation=90)
        ax.set_yticklabels(correlation_matrix.columns)

        for i in range(len(correlation_matrix.columns)):
            for j in range(len(correlation_matrix.columns)):
                ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                        ha="center", va="center")

        ax.set_title("Correlation Matrix", pad=20)

        plt.savefig(savefig, bbox_inches='tight', dpi=300)
        plt.close()

    if savecsv:
        check_path(savecsv)
        correlation_matrix.to_csv(savecsv)

    return styled_matrix
