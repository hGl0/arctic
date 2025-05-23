import numpy as np
from typing import List, Tuple

from sklearn import clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from arctic.models.validation import check_unfitted_model
from .metrics import within_cluster_dispersion
from .sampling import generate_reference_HPPP

def silhouette_method(X: np.ndarray, k_max: int, **kwargs) -> List[float]:
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

def elbow_method(X: np.ndarray, k_max: int, **kwargs) -> Tuple[List[float], List[float]]:
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
    verbose = kwargs.get('verbose', False)

    # check if model is fitted, does not support n_clusters or fit_predict
    check_unfitted_model(model)

    inertias = []
    distortions = []

    for k in range(1, k_max + 1):
        model = clone(model).set_params(n_clusters=k)
        if verbose: print(f"Using {model} as model")

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


def gap_statistic(X: np.ndarray, k_max: int, n_replicates: int = 20, **kwargs) -> np.ndarray:
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
    verbose = kwargs.get('verbose', False)

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

        if verbose: print(f"Using {clustering} as model")
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
