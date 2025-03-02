# necessary libraries
# could be improved?
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plot import *

# CONSTANTS
# 0°C in K
ZERO_DEG = 273.15

# specific gas constant for dry air (J/(kg*K), with 28.964917 g/mol
R = 287.052874

# specific gas constant used by U.S. Standard Atmosphere (J/(K*mol)
# not recommended to use due to inconsistency w.r.t. Avogadro constant and Boltzmann constant
R_ussa = 8.31432

# acceleration due to gravity (m/s^2)
g = 9.80665

# Pressure P_0 at sea level (hPa)
P0 = 1013.25


# compute pca for a given data set
# extract most important features as dataframe
# if plot == True plot the components as vectors vs. the data set
# with x and y being the 2 most important features
def compute_pca(df, comp=4, **kwargs):
    plot_type = kwargs.get('plot_type', '2D')
    scaler = kwargs.get('scaler', StandardScaler)
    savefig = kwargs.get('savefig', None)
    savecsv = kwargs.get('savecsv', None)

    # Scale data with StandardScaler x = (z-u)/s with u being the mean and s the standard deviation
    if scaler:
        try:
            # Check if scaler is a class, not an instance
            if isinstance(scaler, type):
                scaler = scaler()
            scaler.fit(df)
            X = scaler.transform(df)
        except TypeError as e:
            print(f'Type Error: {e}. \n Ensure your dataframe has only numeric types.')
            return None

    # compute PCA
    try:
        pca = PCA()
        pca.fit(X)
        x_new = pca.transform(X)
    except Exception as e:
        print(f"Error while transforming X to PCA: {e}")
        return None

    if plot_type:
        plot_pca(pca, x_new, df, savefig=savefig, plot_type=plot_type)

    # generate overview of influencial features on pca
    scores = pd.DataFrame(pca.components_[:comp].T,
                          columns=[f'PC{i}' for i in range(comp)],
                          index=df.columns)
    # store in csv
    if savecsv:
        scores.to_csv(savecsv)
    return scores
