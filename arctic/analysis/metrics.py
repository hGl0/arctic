import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from statsmodels.tsa.stattools import acf, pacf


def within_cluster_dispersion(X: np.ndarray, labels: np.ndarray, **kwargs) -> float:
    r"""
    Computes the within-cluster dispersion for evaluating clustering compactness.

    :param X: Input data.
    :type X: numpy.ndarray
    :param labels: Cluster labels for each sample.
    :type labels: numpy.ndarray

    :raises ValueError: If `X` and `labels` have mismatched dimensions.

    :return: float representing within-cluster dispersion value.
    :rtype: float
    """
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise TypeError(f"X must be a NumPy array or pandas DataFrame, got {type(X)}")
    if not isinstance(labels, (np.ndarray, pd.Series)):
        raise TypeError(f"labels must be a NumPy array or pandas Series, got {type(labels)}")

    X = np.asarray(X)
    labels = np.asarray(labels)

    # check dimensions
    if len(X) != len(labels):
        raise ValueError("Mismatched dimensions of input data 'X' and 'labels'.")

    Wk = 0
    unique_clusters = np.unique(labels)

    for cluster in unique_clusters:
        cluster_points = X[labels == cluster]
        n_m = len(cluster_points)

        # at least 2 points needed
        if n_m > 1:
            D_m = np.sum(pdist(cluster_points, metric='sqeuclidean'))  # euclidean used in comparative gap statistic
            Wk += D_m / (2 * n_m)
    return Wk
