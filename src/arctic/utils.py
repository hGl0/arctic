import os.path
import numpy as np
from typing import List, Tuple, Optional

import pandas as pd


def check_path(save_path: str) -> None:
    """
    Ensures that a given path is valid and exists. Raises an error otherwise.

    :param save_path: The file path where data will be saved.

    :raises TypeError: If 'save_path' is not a string.
    :raises FileNotFoundError: If the directory does not exist.

    :return: None
    """
    if not isinstance(save_path, str):
        raise TypeError(f"Expected a string (existing file path), but got {type(save_path).__name__}.")

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        raise FileNotFoundError(f"Path '{save_path}' does not exist.\n"
                                f"Please create it before saving, or give a valid path.")


def compute_arrows(pca,
                   score: np.ndarray,
                   n_arrows: int,
                   scales: List[float],
                   n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes and scales the top `n_arrows` vectors for a PCA biplot.

    :param pca: Fitted PCA object containing the principal components.
    :param score: A NumPy array of PCA scores.
    :param n_arrows: The number of top features (arrows) to select.
    :param scales: A list of scaling factors for each principal component.
    :param n: The number of principal components to consider.

    :raises ValueError: If `pca` does not contain 'components_' or
        if `score` has fewer than `n_components` columns.

    :return: A tuple containing:
        - An array of indices corresponding to the selected features.
        - A NumPy array of scaled arrows.
    """
    if not hasattr(pca, "components_"):
        raise ValueError("The provided PCA object must have 'components_' attribute.")
    if score.shape[1] < n:
        raise ValueError(f"Score matrix must have at least {n} columns")

    coeff = pca.components_[:n].T

    # find 'n_arrows' longest arrows
    tops = (coeff ** 2).sum(axis=1).argsort()[-n_arrows:]
    arrows = coeff[tops, :n]

    # Scale arrows
    norm = np.sqrt((arrows ** 2)).sum(axis=0)
    norm[norm == 0] = 1  # avoid division by zero
    arrows /= norm
    arrows *= np.abs(score[:, :n]).max(axis=0)
    scaled_arrows = arrows * np.array(scales[:n])

    return tops, scaled_arrows


def feature_consistence(n_features: Optional[int],
                        features: Optional[List[str]],
                        df: pd.DataFrame) -> Tuple[int, List[str]]:
    """
    Determines the most influential PCA features or uses a given feature list.

    :param n_features: The number of influential features to select.
    :param features: A list of predefined features (optional).
    :param df: A pandas DataFrame containing the dataset.

    :raises ValueError: If neither `n_features` nor `features` is provided.
    :raises Exception: If the loop exceeds the number of available features in `df`.

    :return: A tuple containing:
        - The number of selected features.
        - A list with the feature names
    """

    if features:
        return len(features), features

    if n_features is None:
        raise ValueError("Either 'n_features' or 'features' must be provided.")

    if n_features == df.shape[1]:
        return n_features, df.columns

    # circular import workaround
    from .computation import compute_pca

    add_features = 0
    max_attempts = df.shape[1]

    while True:
        if add_features > max_attempts:
            raise Exception(f"Could not select exactly {n_features} unique features from PCA:\n"
                            f"Tried increasing but reached dataset limit.")
        pca_features = (compute_pca(df, plot_type=None,
                                    comp=n_features+add_features)
                        .drop(['Expl_var', 'Expl_var_ratio'])
                        .idxmax()
                        .reset_index())

        unique_features = pca_features[0].unique()

        if len(unique_features) == n_features:
            return n_features, list(unique_features)
        add_features += 1


# Skips all lines starting with C
# Reads only lines starting with D
def read_data(input_file, enc="utf-8"):
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


# delete space in front of strings
def no_white_space(df, sep=' '):
    col = []
    for c in df.columns:
        col.append(c.split(sep)[-1])
    df.columns = col


# convert strings to actual dates
def to_date(df, col, format='mixed'):
    df[col] = pd.to_datetime(df[col], format=format)