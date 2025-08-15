import warnings
from typing import Tuple, Any

import numpy as np
import pandas as pd
from numpy import ndarray, dtype
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from vortexclust.io.paths import check_path


def compute_pca(df: pd.DataFrame, n_components: int = 4, **kwargs) -> Tuple[np.ndarray, pd.DataFrame, PCA]:
    r"""
    Computes Principal Component Analysis (PCA) for a given dataset.

    :param df: Input data containing numerical data.
    :type df: pandas.DataFrame
    :param n_components: Number of principal components to retain.
    :type n_components: int
    :param kwargs:
        `scaler` (object or type, optional): A
         scaler instance or class. Default is StandardScaler. If no scaler should be used, `scaler`has to be set explicitly to `None`
        `savecsv` (str, optional): File path to save PCA scores.

    :raises ValueError: If `comp` exceeds the number of input features in `df`.
    :raises TypeError: If `df` is not a Pandas DataFrame or contains non-numeric data.
    :raises FileNotFoundError: If `savefig` or `savecsv` directories do not exist.

    :return: Tuple of np.ndarray containing the transformed data,
    a DataFrame containing principal component loadings and explained variance statistics and pca.
    :rtype: Tuple[numpy.ndarray, pandas.DataFrame, PCA object]
    """
    # plot_type = kwargs.get('plot_type', '2D')
    scaler = kwargs.get('scaler', StandardScaler)
    savecsv = kwargs.get('savecsv', None)

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a Pandas DataFrame.")

    if 'label' in df.columns:
        data = df.drop('label', axis=1)
    else:
        data = df

    try:
        # Check if scaler is a class, not an instance
        if callable(scaler):
            scaler = scaler()

            scaler.fit(data)
            X = scaler.transform(data)
        else:
            X = data
    except TypeError as e:
        raise TypeError(f'Type Error: {e}. \n Ensure your dataframe has only numeric types.')

    # compute PCA
    try:
        pca = PCA()
        pca.fit(X)
        x_new = pca.transform(X)
    except Exception as e:
        raise Exception(f"Error while transforming X to PCA: {e}")

    if n_components > pca.components_.shape[1]:
        warnings.warn(UserWarning(f'Less components than given. Reset n_comp to {pca.components_.shape[1]}'))
        n_components = pca.components_.shape[1]

    # generate overview of influence of each features on each principal component
    scores = pd.DataFrame(pca.components_[:n_components].T,
                          columns=[f'PC{i}' for i in range(n_components)],
                          index=data.columns)

    expl_var_row = pd.DataFrame([pca.explained_variance_[:n_components], pca.explained_variance_ratio_[:n_components]],
                                columns=[f"PC{i}" for i in range(n_components)],
                                index=['Expl_var', 'Expl_var_ratio'])
    scores = pd.concat([scores, expl_var_row])

    # store in csv
    if savecsv:
        check_path(savecsv)
        scores.to_csv(savecsv)

    return x_new, scores, pca

def compute_eeof(signal: pd.DataFrame, M: int=400, n_components: int=9) -> tuple[
    Any, Any, Any, ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]]]:
    r"""
    :param signal: Time series of feature.
    :type signal: pandas.DataFrame
    :param M: Sliding window, that should be applied to compute the delay matrix and empirical principal components.
    :type M: int
    :param n_components: Number of principal components to retain.
    :type n_components: int

    :raises TypeError: If `M` is not an integer.
    :raises ValueError: If `M` is greater than the number of time series.

    :return: empirical principal components (EPC), extended empirical orthogonal functions (EEOF),
    eigenvalues explained variance ratios, matrix with reconstructed signal, delay matrix.
    """
    n = len(signal)
    if n < M:
        raise ValueError("Time series length must be greater than window size M.")
    if not isinstance(M, int):
        raise TypeError("M is not an integer.")

    # Build the delay-embedded matrix
    delay_matrix = sliding_window_view(signal, window_shape=M)

    # Apply PCA to the delay matrix
    pca = PCA(n_components=n_components)
    epcs = pca.fit_transform(delay_matrix)
    eeofs = pca.components_
    expl_var_ratio = pca.explained_variance_ratio_

    # Reconstruct the signal using the first n_components
    reconstructed = pca.inverse_transform(epcs)
    # reconstructed = delay_matrix_recon.mean(axis=1)

    # Pad reconstructed to match original length
    full_reconstructed = np.full((n,M), np.nan)
    full_reconstructed[M - 1:] = reconstructed

    return epcs, eeofs, expl_var_ratio, full_reconstructed, delay_matrix


def multivariate_autocorrelation(X, lag=1):
    r"""
    Computes the autocorrelation for a multivariate time series.
    :param X: Time series of features
    :type X: pandas.DataFrame
    :param lag: Lag parameter
    :type lag: int

    :return: Matrix with autocorrelation for each feature
    :rtype: numpy.ndarray
    """
    X = pd.DataFrame(X)
    return [X[col].autocorr(lag=lag) for col in X.columns]