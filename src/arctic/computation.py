import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .plot import *

# compute pca for a given data set
# extract most important features as dataframe
# if plot == True plot the components as vectors vs. the data set
# with x and y being the 2 most important features
def compute_pca(df, comp=4, **kwargs):
    plot_type = kwargs.get('plot_type', '2D')
    scaler = kwargs.get('scaler', StandardScaler)
    n_arrows = kwargs.get('n_arrows', 4)
    savefig = kwargs.get('savefig', None)
    savecsv = kwargs.get('savecsv', None)

    if 'label' in df.columns:
        data = df.drop('label', axis=1)
    else:
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
        plot_pca(pca, x_new, df,
                      savefig=savefig,
                      plot_type=plot_type,
                      n_arrows=n_arrows)

    # generate overview of influence of each features on each principle component
    scores = pd.DataFrame(pca.components_[:comp].T,
                          columns=[f'PC{i}' for i in range(comp)],
                          index=data.columns)

    expl_var_row = pd.DataFrame([pca.explained_variance_[:comp], pca.explained_variance_ratio_[:comp]],
                                columns=[f"PC{i}" for i in range(comp)],
                                index=['Expl_var', 'Expl_var_ratio'])
    scores = pd.concat([scores, expl_var_row])

    # store in csv
    if savecsv:
        arctic.utils.check_path(savecsv)
        scores.to_csv(savecsv)

    return scores