import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from arctic.visualization.pca import plot_pca
from arctic.io.paths import check_path


def compute_pca(df: pd.DataFrame, comp: int = 4, **kwargs) -> pd.DataFrame:
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
    return pca, scores
