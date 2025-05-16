import warnings
from typing import List, Tuple, Optional, Dict, Any, Union, Callable

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import pyproj

from .plot import plot_pca
from .utils import check_path, check_unfitted_model


def compute_pca(df, comp=4, **kwargs):
    r"""
    Computes Principal Component Analysis (PCA) for a given dataset.

    :param df: Input data containing numerical data.
    :type df: pandas.DataFrame
    :param comp: Number of principal components to retain.
    :type comp: int
    :param kwargs:
        `plot_type` (str, optional): '2D' or '3D' plot type. Default is '2D'.
        `scaler` (object or type, optional): A scaler instance or class. Default is StandardScaler.
        `n_arrows` (int, optional): Number of principal component vectors to display. Default is 4.
        `savefig` (str, optional): File path to save the plot.
        `savecsv` (str, optional): File path to save PCA scores.

    :raises ValueError: If `comp` exceeds the number of input features in `df`.
    :raises TypeError: If `df` is not a Pandas DataFrame or contains non-numeric data.
    :raises FileNotFoundError: If `savefig` or `savecsv` directories do not exist.

    :return: DataFrame containing principal component loadings and explained variance statistics.
    :rtype: pandas.DataFrame
    """
    plot_type = kwargs.get('plot_type', '2D')
    scaler = kwargs.get('scaler', StandardScaler)
    n_arrows = kwargs.get('n_arrows', 4)
    savefig = kwargs.get('savefig', None)
    savecsv = kwargs.get('savecsv', None)

    if 'label' in df.columns:
        labels = df['label']
        data = df.drop('label', axis=1)
    else:
        labels = None
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

    # compute PCA
    try:
        pca = PCA()
        pca.fit(X)
        x_new = pca.transform(X)
    except Exception as e:
        raise Exception(f"Error while transforming X to PCA: {e}")

    if plot_type:
        plot_pca(pca, x_new,
                 features=df.columns.tolist(), labels=labels,
                 savefig=savefig,
                 plot_type=plot_type,
                 n_arrows=n_arrows)

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
# only scores really usefull? Maybe also transformed dataset?
    return scores


def generate_reference_HPPP(X):
    r"""
    Generates a reference dataset using a Homogeneous Poisson Point Process (HPPP).

    :param X: Input data.
    :type X: numpy.ndarray

    :raises ValueError: If `X` contains non-numeric values.

    :return: ndarray containing a randomly generated reference data matching the shape and range of 'X'
    :rtype: numpy.ndarray
    """
    # check if X is numeric
    return np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=X.shape)


def within_cluster_dispersion(X, labels, **kwargs):
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


def gap_statistic(X, k_max, n_replicates=20, **kwargs):
    r"""
    Computes the Gap Statistic to determine the optimal number of clusters.

    :param X: Input data.
    :type X: numpy.ndarray
    :param k_max: Maximum number of clusters to evaluate.
    :type k_max: int
    :param n_replicates: Number of reference datasets to generate for each k.
    :type n_replicates: int
    :param kwargs:
        model (object, optional): A clustering model instance (e.g. `KMeans(n_init=10)`).
                                  Must support `fit_predict(X)` and accept `n_clusters` via `set_params`.
        weights (array-like, optional): Not used in current implementation. Reserved for future use.
                                  Sample weights for computing weighted dispersion.



    :raises ValueError: If `k_max` or `n_replicates` is less than 1.
                        If a fitted model is passed instead of an unfitted instance.
    :raises TypeError: If the passed model does accept `n_clusters`

    :return: Array of shape (k_max, 2) containing gap values and standard deviations.
    :rtype: numpy.ndarray
    """
    model = kwargs.get('model', AgglomerativeClustering(linkage='complete'))

    if k_max <= 0:
        raise ValueError(f"Maximum number of clusters to consider should be a positive integer, got {k_max} instead")
    if n_replicates <= 0:
        raise ValueError(
            f"Number of reference data sets to generate should be a positive integer, got {n_replicates} instead")

    # check if model is fitted, does not support n_clusters or fit_predict
    check_unfitted_model(model)

    gap_values = []

    for k in range(1, k_max + 1):
        clustering = clone(model).set_params(n_clusters=k)
        model_ref = clone(model).set_params(n_clusters=k)

        print(f"Using {clustering} as model")
        # Fit model to the original data
        labels = clustering.fit_predict(X)
        log_WK = np.log(within_cluster_dispersion(X, labels))

        # Compute the reference dispersion values
        log_Wk_star = []
        for _ in range(n_replicates):
            ref_data = generate_reference_HPPP(X)  # fixed sample size
            ref_labels = model_ref.fit_predict(ref_data)
            log_Wk_star.append(np.log(within_cluster_dispersion(ref_data, ref_labels)))

        # Compute gap
        gap = np.mean(log_Wk_star) - log_WK

        # Account for simulation error
        s_k = np.std(log_Wk_star) * np.sqrt(1 + 1 / n_replicates)

        gap_values.append((gap, s_k))

    return np.array(gap_values)


def elbow_method(X, k_max, **kwargs):
    r"""
    Computes distortion and inertia metrics to determine the optimal number of clusters using the elbow method.

    :param X: Input data.
    :type X: numpy.ndarray
    :param k_max: Maximum number of clusters to evaluate.
    :type k_max: int
    :param kwargs:
    :param kwargs:
        model (object, optional): A clustering model instance (e.g. `KMeans(n_init=10)`).
                                  Must support `fit_predict(X)` and accept `n_clusters` via `set_params`.
                                  Default is AgglomerativeClustering(linkage='average').
        weights (array-like, optional): Sample weights for computing weighted distortion.

    :raises ValueError: If `k_max` is less than 1.
    :raises TypeError: If `X` is not a NumPy ndarray.

    :return: Two lists: mean distortions and total inertias for each cluster count.
    :rtype: tuple[list[float], list[float]]
    """

    weights = kwargs.get('weights', None)
    model = kwargs.get('model', AgglomerativeClustering(linkage='complete'))

    # check if model is fitted, does not support n_clusters or fit_predict
    check_unfitted_model(model)

    inertias = []
    distortions = []

    for k in range(1, k_max + 1):
        model = clone(model).set_params(n_clusters=k)

        labels = model.fit_predict(X)
        n_total = X.shape[0]
        all_sq_dists = np.zeros(n_total)
        for l in np.unique(labels):
            cluster_points = X[labels == l]
            # compute centroid
            # when X np.array [] around ci, else not?
            ci = np.mean(cluster_points, axis=0)

            squ_dists = np.sum((cluster_points - ci) ** 2, axis=1)
            all_sq_dists[labels == l] = squ_dists

        distortions.append(np.average(all_sq_dists, weights=weights))
        inertias.append(np.sum(all_sq_dists))
    return distortions, inertias


def silhouette_method(X, k_max, **kwargs):
    r"""
    Computes scores for a range of cluster counts using the silhouette method.

    :param X: Input data.
    :type X: numpy.ndarray
    :param k_max: Maximum number of clusters to evaluate (must be ≥ 2).
    :type k_max: int
    :param kwargs:
        `model` (object, optional): Clustering model class with a `fit_predict` method.
                    Default is sklearn's AgglomerativeClustering.
        `labels` (numpy.ndarray, optional): Array with the labels for each k.
                    Has to have shape (n_samples, k_max)

    :raises ValueError: If `k_max` is less than 2.
    :raises TypeError: If `X` is not a NumPy ndarray.

    :return: List of silhouette scores for each number of clusters from 2 to `k_max`.
    :rtype: list[float]
    """
    model = kwargs.get('model', AgglomerativeClustering(linkage='complete'))
    given_labels = kwargs.get('labels', None)

    # if given_labels is not None and given_labels.shape[1] != k_max-1:
    #     raise ValueError("The shape of the given labels needs to be (n_samples, 'k_max')."
    #                      f"The passed labels have shape {given_labels.shape}")

    # check if model is fitted, does not support n_clusters or fit_predict
    check_unfitted_model(model)

    s = []
    if k_max < 2:
        raise ValueError("k_max must be at least 2 for silhouette method.")
    for k in range(2, k_max + 1):
        if given_labels:
            labels = given_labels[k - 2]
        else:
            model = clone(model).set_params(n_clusters=k)
            labels = model.fit_predict(X)

        s.append(silhouette_score(X, labels))
    return s


# maybe in utils?
# projection modification?
def compute_ellipse(area, ar, theta, loncent, latcent,
                    num_points = 200):
    r"""
    Computes the coordinates of a rotated ellipse based on geophysical parameters and
    projects it onto a polar stereographic coordinate system centered near the North Pole.

    :param area: Area of the ellipse in square kilometers.
    :type area: float
    :param ar: Aspect ratio of the ellipse (major axis / minor axis).
    :type ar: float
    :param theta: Orientation angle of the major axis in radiant, measured counter-clockwise from east.
    :type theta: float
    :param loncent: Longitude of the ellipse center in degrees.
    :type loncent: float
    :param latcent: Latitude of the ellipse center in degrees.
    :type latcent: float
    :param num_points: Number of points to use for generating the ellipse perimeter. Default is 200.
    :type num_points: int, optional

    :raises ValueError:
    :raises TypeError:

    :return: Triple containing the projected x and y coordinates (in meters) of the rotated ellipse and projection.
    :rtype: tuple of np.ndarray
    """
    # Calculate semi-major (a) and semi-minor (b) axes
    b = np.sqrt(area / (np.pi * ar))  # Minor axis length [km]
    a = ar * b  # Major axis length [km]

    # Create points for the ellipse in x-y coordinate system
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(t)  # [km]
    y = b * np.sin(t)  # [km]

    # Rotate the ellipse by theta degrees
    theta_rad = theta + np.radians(90)  # radiant, account for 0° in x-y not equal 0° North Polar Projection
    x_rot = x * np.cos(theta_rad) - y * np.sin(theta_rad)  # [km]
    y_rot = x * np.sin(theta_rad) + y * np.cos(theta_rad)  # [km]

    # Define stereographic projection centered on the ellipse
    proj_pyproj = pyproj.Proj(proj='stere',
                              lat_0=90,  # latcent
                              lon_0=0,  # loncent
                              lat_ts=60,  # latcent
                              ellps='WGS84')
    # Convert center to x-y
    x_center, y_center = proj_pyproj(loncent, latcent)

    # Translate ellipse to the center in meters
    x_final = x_center + x_rot * 1000  # if x_rot is in km
    y_final = y_center + y_rot * 1000
    return x_final, y_final, proj_pyproj
