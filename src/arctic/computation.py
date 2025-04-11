import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from .plot import plot_pca
from .utils import *

def compute_pca(df, comp=4, **kwargs):
    """
    Computes Principal Component Analysis (PCA) for a given dataset.

    :param df (pd.DataFrame): Input dataframe containing numerical data.
    :param comp (int): Number of principal components to retain.
    :param kwargs:
        - `plot_type` (str, optional): '2D' or '3D' plot type. Default is '2D'.
        - `scaler` (object, optional): Scaler to normalize data. Default is StandardScaler.
        - `n_arrows` (int, optional): Number of principal component vectors to display. Default is 4.
        - `savefig` (str, optional): File path to save the plot.
        - `savecsv` (str, optional): File path to save PCA scores.

    :raises ValueError: If `comp` is greater than the number of features in `df`.
    :raises TypeError: If `df` is not a Pandas DataFrame.
    :raises FileNotFoundError: If `savefig` or `savecsv` directories do not exist.

    :return: pd.DataFrame containing principal component scores and explained variance.
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
        labels=None
        data = df

    # Scale data with StandardScaler x = (z-u)/s with u being the mean and s the standard deviation
    if scaler:
        try:
            # Check if scaler is a class, not an instance
            if isinstance(scaler, type):
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

    return scores



def generate_reference_HPPP(X):
    """
    Generates a reference dataset using a Homogeneous Poisson Point Process (HPPP).

    :param X: (ndarray) Input data.

    :raises ValueError: If `X` contains non-numeric values.

    :return: ndarray containing a randomly generated reference dataset.
    """
    return np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=X.shape)


def within_cluster_dispsersion(X, labels):
    """
    Computes the within-cluster dispersion for hierarchical clustering.

    :param X: (ndarray) Input data.
    :param labels: (ndarray) Cluster labels for each sample.

    :raises ValueError: If `X` and `labels` have mismatched dimensions.

    :return: float representing within-cluster dispersion value.
    """
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
    """
    Computes the Gap Statistic to determine the optimal number of clusters.

    :param X: (ndarray) Input data.
    :param k_max: (int) Maximum number of clusters to evaluate.
    :param n_replicates: (int) Number of bootstrap samples.
    :param kwargs:
        - `model` (object, optional): Clustering model. Default is AgglomerativeClustering.

    :raises ValueError: If `k_max` or `n_replicates` is less than 1.
    :raises TypeError: If `X` is not a numeric ndarray.

    :return: np.ndarray containing gap values and standard deviations for each k.
    """
    model = kwargs.get('model', AgglomerativeClustering)

    if k_max <= 0:
        raise ValueError(f"Maximum number of clusters to consider should be a positive integer, got {k_max} instead")
    if n_replicates <= 0:
        raise ValueError(
            f"Number of reference data sets to generate should be a positive integer, got {n_replicates} instead")

    gap_values = []

    for k in range(1, k_max + 1):
        # Fit Agglomerative Clustering to the original data
        clustering = AgglomerativeClustering(n_clusters=k, linkage='average')
        labels = clustering.fit_predict(X)
        log_WK = np.log(within_cluster_dispsersion(X, labels))

        # Compute the reference dispersion values
        log_Wk_star = []
        for _ in range(n_replicates):
            ref_data = generate_reference_HPPP(X)  # fixed sample size
            model_ref = AgglomerativeClustering(n_clusters=k, linkage='average')
            ref_labels = model_ref.fit_predict(ref_data)
            log_Wk_star.append(np.log(within_cluster_dispsersion(ref_data, ref_labels)))

        # Compute gap
        gap = np.mean(log_Wk_star) - log_WK

        # Account for simulation error
        s_k = np.std(log_Wk_star) * np.sqrt(1 + 1 / n_replicates)

        gap_values.append((gap, s_k))

    return np.array(gap_values)


def elbow_method(X, k_max, **kwargs):
    weights = kwargs.get('weights', None)

    inertias = []
    distortions = []

    for k in range(1, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
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

        distortions.append(np.mean(all_sq_dists))
        inertias.append(np.sum(all_sq_dists))
    return distortions, inertias


def silhouette_method(X, k_max):
    s = []
    if k_max < 2:
        print("Invalid value for silhouette method")
    for k in range(2, k_max+1):
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = model.fit_predict(X)
        s.append(silhouette_score(X, labels))
    return s