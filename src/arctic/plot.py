# necessary libraries
# could be improved?
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from . import compute_pca
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler


# plots a dendrogram for a given model with computed distances between samples
# model requires to have children_ and distances_ attribute
def plot_dendrogram(model, **kwargs):
    savefig = kwargs.pop('savefig', None)

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

        if savefig:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)

    # Catch errors
    except AttributeError as e:
        print(f"AttributeError: {e}.\n Ensure 'model' has 'children_', 'distances_', and 'labels_' attributes.")
    except TypeError as e:
        print(f"TypeError: {e}.\n Check if model properties are of the correct data type.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# compute correlation coefficient of a dataframe
# recommended to use equally perceived colormaps with 0 as white
# according to https://matplotlib.org/stable/users/explain/colors/colormaps.html
def plot_correlation(df, **kwargs):
    try:
        if not isinstance(df, **kwargs):
            raise TypeError("Expected input 'df' to be a Pandas DataFrame")

        savefig = kwargs.get('savefig', None)
        cmap = kwargs.get('cmap', 'RdBu')
        try:
            correlation_matrix = df.corr(numeric_only=True)
            styled_matrix = correlation_matrix.style.background_gradient(cmap=cmap, vmin=-1, vmax=1.0)
        except AttributeError as e:
            print(f"Attribute Error {e}: Expected 'df' to be a Pandas Dataframe")

        if savefig:
            fig, ax = plt.subplots(figsize=(15, 15))

            ax.matshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1.0)

            ticks = np.arange(len(correlation_matrix.columns))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(correlation_matrix.columns, rotation=90)
            ax.set_yticklabels(correlation_matrix.columns)

            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                            ha="center", va="center")

            ax.set_title("Correlation Matrix", pad=20)

            try:
                if savefig:
                    plt.savefig(savefig, bbox_inches='tight', dpi=300)
                plt.close()
            except FileNotFoundError as e:
                print(f"FileNotFoundError {e}: Could not save the file. Check if the directory exists: {savefig}")

        return styled_matrix
    except Exception as e:
        print(f"An error occurred while generating the correlation matrix: {e}")


# plot the components of a pca as vectors vs. the data set
# with x and y being the 2 most important features
def plot_pca(pca, x_new, df, savefig=None, **kwargs):
    plot_type = kwargs.get('plot_type', '2D')

    # Init values for plotting
    try:
        score = x_new[:, :]
        coeff = np.transpose(pca.components_[:, :])

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
                    ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, coeff[i, 2] * 1.15,
                            'Var' + str(i + 1),
                            color='g', ha='center', va='center')
                else:
                    ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, coeff[i, 2] * 1.15,
                            labels[i],
                            color='g', ha='center', va='center')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.title('3D Plot')
            #save figure
            if savefig:
                plt.savefig(savefig, bbox_inches='tight', dpi=300)
            plt.show()

        else:
            print(f"Warning: Unknown plot type '{plot_type}'")

    except Exception as e:
        print(f"Error while plotting: {e}")


# Plots a radar chart for a given dataframe
# Either n_features or features must be given
# standard value is n_features to select the n most important features from pca
def plot_radar(df, label='label', **kwargs):
    features = kwargs.get('features', None)  # List of features to plot
    n_features = kwargs.get('n_features', 6)  # Use the 6 most important features
    savefig = kwargs.get('savefig', None)  # location to save figure
    agg_func = kwargs.get('agg_func', 'mean')  # aggregation function used with groupby
    scaler = kwargs.get('scaler', StandardScaler)

    if n_features and features:
        raise ValueError("Provide either 'features' (list of columns) "
                         "or 'n_features' (integer), not both. "
                         "If none is provide n_features = 6.")

    # set n_features to the amount of given features by list
    if features:
        n_features = len(features)
    else:
        # magic to get n most important features from pca
        #features = df.columns[:n_features]
        features = abs(compute_pca(df, plot_type=None,
                                      comp=n_features)).idxmax()

    # group and aggregate dataframe
    try:
        # check for custom aggregation function
        grouped = df.groupby(label)
        if callable(agg_func):
            val = grouped.apply(agg_func)
        elif isinstance(agg_func, str) and hasattr(grouped, agg_func):
            val = getattr(grouped, agg_func)()
        else:
            raise ValueError("Invalid 'agg_func'. Use 'mean', 'median', 'sum', or a callable function.")
    except KeyError as e:
        raise KeyError(
            "Please give a valid label or ensure that your Pandas DataFrame contains a column named 'label'.")
    except Exception as e:
        print(f"Unexpected exception while aggregation: {e}")

    group = val.index.tolist()

    # Scale features for nicer look
    try:
        if isinstance(scaler, type):
            scaler = scaler()
        val_scaled = pd.DataFrame(scaler.fit_transform(val), columns=val.columns)
    except TypeError as e:
        raise TypeError('Ensure your dataframe has only numeric types.')
    except Exception as e:
        print(f"Error while scaling: {e}")

    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    for g in group:
        values = val_scaled.loc[g, features].tolist()
        values += values[:1]
        ax.plot(angles, values, marker='o', linewidth=2, label=f"{g}")
        ax.fill(angles, values, alpha=0.3)

        for i, (angle, orig_val) in enumerate(zip(angles[:-1], val.loc[g, features])):
            if orig_val > 1000:
                ax.text(angle, values[i]+0.05, f"{orig_val:.2e}",
                        va='center', ha='center',
                        fontsize=8)
            else:
                ax.text(angle, values[i] + 0.05, f"{orig_val:.2f}",
                    ha='center', va='center',
                    fontsize=8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{f}" for f in features])
    ax.set_yticklabels([])

    plt.legend(title='Cluster')
    plt.title(f"Radar Chart")

    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
    plt.show()
    return
