# necessary libraries
# could be improved?
import os.path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
import warnings


def check_path(save_path):
    """Ensures that a given path exists and is valid, i.e. a string. Raises an error otherwise."""
    if save_path is None:
        raise ValueError("Expected 'save_path' to be a valid file path, but got None.")

    if not isinstance(save_path, str):
        raise TypeError(f"Expected 'savefig' to be string (existing file path), but got {type(save_path).__name__}.")

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        raise FileNotFoundError(f"Path '{save_path}' does not exist.\n"
                                f"Please create it before saving, or give a valid path.")


def plot_dendrogram(model, **kwargs):
    """
    Plots a dendrogram for a given hierarchical clustering model with computed distances between samples.

    :param model: A clustering model that must have 'children_', 'distances_', and 'labels_' attributes.
    :param kwargs: Additional arguments for 'scipy.cluster.hierarchy.dendrogram'.
                - `savefig` (str, optional): Path to save the figure. Raises an error if the directory does not exist.

    :raises AttributeError: If 'model' is missing required attributes.
    :raises TypeError: If 'model' attributes are of incorrect data types.
    :raises FileNotFoundError: If the directory for `savefig` does not exist.
    :raises Exception: For unexpected errors.

    :return: None
    """

    save_path = kwargs.pop('savefig', None)

    # check availability of attributes
    if not all(hasattr(model, attr) for attr in ['children_', 'distances_', 'labels_']):
        raise AttributeError("Model must have 'children_', 'distances_', and 'labels_' attributes.")


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

        # save the dendrogram
        if save_path:
            # Validate savefig
            check_path(save_path)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

    # Catch errors
    except AttributeError as e:
        raise AttributeError(f"Model is missing required attributes: {e}")
    except TypeError as e:
        warnings.warn(f"TypeError: {e}.\n Ensure 'children_' and 'distances_' are of correct type.")
    except FileNotFoundError as e:
        raise e
    except Exception as e:
        warnings.warn(f"Unexpected error: {e}")


def plot_correlation(df, **kwargs):
    """
    Plots a correlation matrix for a given DataFrame.
    Recommended to use equally perceived colormaps with 0 as white, according to https://matplotlib.org/stable/users/explain/colors/colormaps.html
    :param df: Input pandas DataFrame
    :param kwargs:
                - savefig (str, optional): File path to save the correlation matrix as picture. Default is None.
                - cmap (str, optional): Colormap for visualisation. Default is 'RdBu'.
    :raises TypeError: If the input DataFrame is not a pandas DataFrame, or 'savefig' is not a string.
    :raises FileNotFoundError: If 'savefig' is not a valid path.
    :raises AttributeError: If the input DataFrame is not a pandas DataFrame.
    :return: pd.io.formats.style.Styler: Styled correlation matrix for Jupyter display, or None if an error occurs.
    """

    save_path = kwargs.get('savefig', None)
    cmap = kwargs.get('cmap', 'RdBu')

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected input 'df' to be a Pandas DataFrame")

    if df.empty:
        warnings.warn("The provided Dataframe is empty. Returning None.")
        return None

    if not isinstance(cmap, str) or cmap not in plt.colormaps():
        warnings.warn(f"Invalid colormap '{cmap}'. Defaulting to 'RdBu'.")
        cmap = 'RdBu'

    try:
        correlation_matrix = df.corr(numeric_only=True)
        try:
            styled_matrix = correlation_matrix.style.background_gradient(cmap=cmap, vmin=-1, vmax=1.0)
        except AttributeError as e:
            raise AttributeError(f"Attribute Error {e}: Expected 'df' to be a Pandas DataFrame")

        if save_path:
            # validate save_path
            check_path(save_path)

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

            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

        return styled_matrix
    except Exception as e:
        raise Exception(f"An error occurred while generating the correlation matrix: {e}")


# plot the components of a pca as vectors vs. the data set
# with x and y being the 2 most important features
def plot_pca(pca, x_reduced, df, savefig=None, **kwargs):
    plot_type = kwargs.get('plot_type', '2D')
    n_arrows = kwargs.get('n_arrows', 4)

    # Init values for plotting
    try:
        score = x_reduced[:, :]
        pvars = pca.explained_variance_ratio_ * 100

        xs = score[:, 0]
        ys = score[:, 1]
        zs = score[:, 2]

        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())
        scalez = 1.0 / (zs.max() - zs.min())

        features = getattr(df, 'columns', None)

        # color data by label
        c = df['label'].to_list() if 'label' in df else None
        cmap = ListedColormap(plt.get_cmap('tab10').colors[:len(np.unique(c))]) if c else None

    except Exception as e:
        print(f"Error while assigning values: {e}")
        return None

    # Plot data and principal components
    # decision again match-case statement, might not work in older python versions (< 3.10)
    try:
        if plot_type == '2D':
            coeff = pca.components_[:2].T
            # generate arrows and scale them
            # 1. approach: find n_arrows longest arrows
            tops = (coeff ** 2).sum(axis=1).argsort()[-n_arrows:]
            arrows = coeff[tops, :2]

            # 2. approach: find n_arrows features that drive most variance in the visible pcs
            # tops = (loadings*pvars).sum(axis=1).argsort()[-n_arrows:]
            # arrows = loadings[tops]

            # Scale arrows
            arrows /= np.sqrt((arrows ** 2)).sum(axis=0)
            arrows *= np.abs(score[:, :2]).max(axis=0)
            scaled_arrows = arrows * np.array([scalex, scaley])

            fig = plt.figure()
            ax = fig.add_subplot(111)

            if c is not None:
                # create mask
                for i, label in enumerate(np.unique(c)):
                    mask = np.array(c) == label
                    ax.scatter(xs[mask] * scalex, ys[mask] * scaley,
                               color=cmap(i), label=label, alpha=0.5, zorder=0)
                ax.legend(title='Cluster')

            else:
                ax.scatter(xs * scalex, ys * scaley, alpha=0.5, zorder=0)

            # new arrows
            for i, arrow in zip(tops, scaled_arrows):
                ax.quiver(0, 0, *arrow, color='gray',
                          zorder=3,
                          angles='xy', scale_units='xy',
                          scale=1)

                if features is None:
                    ax.text(*(arrow * 1.15),
                            'Var' + str(i + 1),
                            color='g', ha='center', va='center')
                else:
                    ax.text(*(arrow * 1.15), features[i], ha='center', va='center')

            for i, axis in enumerate('xy'):
                getattr(ax, f'set_{axis}label')(f'PC{i + 1} ({pvars[i]:.2f}%)')

            # make up for plot
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            plt.grid()
            plt.title('2D Biplot')

        elif plot_type == '3D':
            coeff = pca.components_[:3].T
            # generate arrows and scale them
            # 1. approach: find n_arrows longest arrows
            tops = (coeff ** 2).sum(axis=1).argsort()[-n_arrows:]
            arrows = coeff[tops, :3]

            # 2. approach: find n_arrows features that drive most variance in the visible pcs
            # tops = (loadings*pvars).sum(axis=1).argsort()[-n_arrows:]
            # arrows = loadings[tops]

            # Scale arrows
            arrows /= np.sqrt((arrows ** 2)).sum(axis=0)  # float errors!!
            arrows *= np.abs(score[:, :3]).max(axis=0)  # adjust to score
            scaled_arrows = arrows * np.array([scalex, scaley, scalez])  # same scale as points
            len_arrows = np.sqrt(((scaled_arrows) ** 2).sum(axis=1))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for i, arrow in zip(tops, scaled_arrows):
                ax.quiver(0, 0, 0, *arrow, color='gray',
                          zorder=3)

                if features is None:
                    ax.text(*(arrow * 1.15),
                            'Var' + str(i + 1),
                            color='g', ha='center', va='center')
                else:
                    ax.text(*(arrow * 1.15), features[i], ha='center', va='center')

            # plot points
            if c is not None:
                scatter = ax.scatter3D(xs * scalex, ys * scaley, zs * scalez,
                                       alpha=0.5, c=c, cmap='tab10', zorder=0)
                legend_handles, legend_labels = scatter.legend_elements(prop="colors")
                ax.legend(legend_handles, legend_labels, title="Cluster")
            else:
                ax.scatter3D(xs * scalex, ys * scaley, zs * scalez, alpha=0.5)

            for i, axis in enumerate('xyz'):
                getattr(ax, f'set_{axis}label')(f'PC{i + 1} ({pvars[i]:.2f}%)')

            # make up for plot
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.grid()
            plt.title('3D Biplot')
        else:
            print(f"Warning: Unknown plot type '{plot_type}'")

        # save figure
        if savefig:
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
        plt.show()

    except Exception as e:
        print(f"Error while plotting: {e}")


# Plots a radar chart for a given dataframe
# Either n_features or features must be given
# standard value is n_features to select the n most important features from pca
def plot_radar(df, label='label', **kwargs):
    features = kwargs.get('features', None)  # List of features to plot
    n_features = kwargs.get('n_features', 6)  # Use the 6 most important features
    savefig = kwargs.get('savefig', None)  # location to save figur
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
        features = df.columns[:n_features]
        # features = abs(compute_pca(df, plot_type=None,
        #                              comp=n_features)).idxmax()

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
                ax.text(angle, values[i] + 0.05, f"{orig_val:.2e}",
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


# Violin plot for each cluster to compare their characteristics
def plot_violin(df, label='label', **kwargs):
    pass
