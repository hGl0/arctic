import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# from arctic.visualization.pca import plot_pca
from arctic.io.paths import check_path


def compute_pca(df: pd.DataFrame, n_comp: int = 4, **kwargs) -> Tuple[np.ndarray, pd.DataFrame, PCA]:
    r"""
    Computes Principal Component Analysis (PCA) for a given dataset.

    :param df: Input data containing numerical data.
    :type df: pandas.DataFrame
    :param n_comp: Number of principal components to retain.
    :type n_comp: int
    :param kwargs:
        `scaler` (object or type, optional): A scaler instance or class. Default is StandardScaler.
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

    if 'label' in df.columns:
        data = df.drop('label', axis=1)
    else:
        data = df

    # Scale data with StandardScaler x = (z-u)/s with u being the mean and s the standard deviation
    if scaler:
        try:
            # Check if scaler is a class, not an instance
            if callable(scaler):
                scaler = scaler()

            scaler.fit(data)
            X = scaler.transform(data)
        except TypeError as e:
            raise TypeError(f'Type Error: {e}. \n Ensure your dataframe has only numeric types.')
    else:
        X = data

    # compute PCA
    try:
        pca = PCA()
        pca.fit(X)
        x_new = pca.transform(X)
    except Exception as e:
        raise Exception(f"Error while transforming X to PCA: {e}")

    # removed for clearer structure
    # if plot_type:
    #     plot_pca(pca, x_new,
    #              features=df.columns.tolist(), labels=labels,
    #              savefig=savefig,
    #              plot_type=plot_type,
    #              n_arrows=n_arrows)

    # generate overview of influence of each features on each principal component
    scores = pd.DataFrame(pca.components_[:comp].T,
                          columns=[f'PC{i}' for i in range(comp)],
                          index=data.columns)

    expl_var_row = pd.DataFrame([pca.explained_variance_[:comp], pca.explained_variance_ratio_[:comp]],
                                columns=[f"PC{i}" for i in range(comp)],
                                index=['Expl_var', 'Expl_var_ratio'])
    scores = pd.concat([scores, expl_var_row])

    # store in csv
    if savecsv:
        check_path(savecsv)
        scores.to_csv(savecsv)

    return x_new, scores, pca

def compute_eeof(signal, M=400, n_components=9):
    r"""
    :param signal:
    :param M:
    :param n_components:

    :raises ValueError: If `M` is not an integer or greater than the number of time series.

    :return:
    """
    n = len(signal)
    if n < M:
        raise ValueError("Time series length must be greater than window size M.")

    # Build the delay-embedded matrix
    delay_matrix = sliding_window_view(signal, window_shape=M)

    # Apply PCA to the delay matrix
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(delay_matrix)
    eigenvalues = pca.explained_variance_

    # Reconstruct the signal using the first n_components
    reconstructed = pca.inverse_transform(pcs)
    # reconstructed = delay_matrix_recon.mean(axis=1)

    # Pad reconstructed to match original length
    full_reconstructed = np.full((n,M), np.nan)
    full_reconstructed[M - 1:] = reconstructed

    return full_reconstructed, pca, delay_matrix, eigenvalues
