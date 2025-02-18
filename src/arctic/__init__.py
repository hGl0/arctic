# necessary libraries
# could be improved?
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

##### Constants #####
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


# plots a dendrogram for a given model with computed distances between samples
# model requires to have children_ and distances_ attribute
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    try:
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]
        ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    # Catch errors
    except AttributeError as e:
        print(f"AttributeError: {e}.\n Ensure 'model' has 'children_', 'distances_', and 'labels_' attributes.")
    except TypeError as e:
        print(f"TypeError: {e}.\n Check if model properties are of the correct data type.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# compute pca for a given data set
# extract most important features as dataframe
# if plot == True plot the components as vectors vs. the data set
# with x and y being the 2 most important features
def pca_importance(df, comp=4, **kwargs):
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

    # Init values for plotting
    try:
        score = x_new[:, :]
        coeff = np.transpose(pca.components_[:, :])  # voher 0:comp, :

        xs = score[:, 0]
        ys = score[:, 1]
        zs = score[:, 2]

        n = coeff.shape[0]

        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        scalez = 1.0 / (zs.max() - zs.min())

        labels = df.columns.to_list() if hasattr(df, 'columns') else None
        # color data by label
        if 'label' in df.columns:
            c = df['label'].to_list()
        else:
            c = None
    except Exception as e:
        print(f"Error while assigning values: {e}")
        return None

    # Plot data and principal components
    # decision again match-case statement, might not work in older python versions (< 3.10)
    try:
        if plot_type == '2D':
            if c:
                plt.scatter(xs * scalex, ys * scaley, c=c, cmap='tab10')  #maybe useful with cluster labels?
            else:
                plt.scatter(xs * scalex, ys * scaley)
            # Plot arrows of features
            for i in range(n):
                plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
                if labels is None:
                    plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                             "Var" + str(i + 1),
                             color='g', ha='center', va='center')
                else:
                    plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                             labels[i],
                             color='g', ha='center', va='center')

            # make up for plot
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.xlabel("PC{}".format(1))
            plt.ylabel("PC{}".format(2))
            plt.grid()
            plt.title('2D Plot')
            # save figure
            if savefig: plt.savefig(savefig)
            plt.show()

        elif plot_type == '3D':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # Plot of pca
            if c:
                ax.scatter3D(xs * scalex, ys * scaley, zs * scalez, alpha=0.5, c=c, cmap='tab10')
            else:
                ax.scatter3D(xs * scalex, ys * scaley, zs * scalez, alpha=0.5)
            # Plot arrows of features
            for i in range(n):
                ax.quiver(0, 0, 0, coeff[i, 0], coeff[i, 1], coeff[i, 2], color='r', alpha=0.5)
                if labels is None:
                    ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, coeff[i, 2]*1.15,
                            'Var' + str(i + 1),
                            color='g', ha='center', va='center')
                else:
                    ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, coeff[i,2]*1.15,
                            labels[i],
                            color='g', ha='center', va='center')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.title('3D Plot')
            #save figure
            if savefig: plt.savefig(savefig)
            plt.show()
        else:
            print(f"Warning: Unkown plot type '{plot_type}'")
    except Exception as e:
        print(f"Error while plotting: {e}")

    # generate overview of influencial features on pca
    scores = pd.DataFrame(pca.components_[:comp].T,
                          columns=[f'PC{i}'.format(i) for i in range(comp)],
                          index=df.columns)
    # store in csv
    if savecsv: scores.to_csv(savecsv)
    return scores


# compute correlation coefficient of a dataframe
# recommended to use equally perceived colormaps with 0 as white
# according to https://matplotlib.org/stable/users/explain/colors/colormaps.html
def plot_correlation(df, cmap='RdBu'):
    correlation_matrix = df.corr(numeric_only=True)
    return correlation_matrix.style.background_gradient(cmap=cmap, vmin=-1, vmax=1.0)
