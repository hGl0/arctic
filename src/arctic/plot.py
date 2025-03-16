# necessary libraries
# could be improved?
from .utils import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
import warnings


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


def plot_pca(pca, x_reduced,
             features=None,
             labels=None,
             savefig=None,
             **kwargs):
    """
    Plots a 2D or 3D Principal Component Analysis (PCA) biplot, visualizing the dataset
    and the most important principal component vectors.

    :param pca: A fitted PCA object from sklearn.decomposition.PCA.
        Must contain `components_` and `explained_variance_ratio_` attributes.
    :param x_reduced: ndarray of shape (n_samples, n_components). The dataset transformed into principal components.
    :param features: List of feature names correpsonding to the original dataset.
        Used to label principal component vectors. Default is None.
    :param labels: List of cluster labels for coloring data points with its respective cluster. Default is None.
    :param savefig: (str, optional) File path to save the plot. Default is None.
    :param kwargs:
                - plot_type (str, optional): '2D' or '3D' plot. Default is '2D'
                - n_arrows (int, optional): Number of principal component vectors to display. Default is 4.

    :raises AttributeError: If `pca` is missing required attributes.
    :raises TypeError: If input types are incorrect.
    :raises ValueError: If `plot_type` is not '2D' or '3D'.
    :raises FileNotFoundError: If `savefig` path is invalid.

    :return: None
    """
    plot_type = kwargs.get('plot_type', '2D')
    n_arrows = kwargs.get('n_arrows', 4)

    # Init values for plotting
    try:
        score = x_reduced[:, :]
        pvars = pca.explained_variance_ratio_ * 100

        xs, ys, zs = score[:, :3].T

        scales = 1.0 / (score.max(axis=0) - score.min(axis=0))
        scalex, scaley, scalez = scales[:3]

        # color by label
        if isinstance(labels, pd.Series):
            if labels.nunique() > 10:
                warnings.warn(f"Currently using 'tab10' to assign colors for each cluster.\n"
                              f"There are {labels.nunique()} unique labels, so the plot might not be displayed correctly.\n"
                              f"Working on this issue.")
            c = labels.to_list()
        else:
            c = None

    except Exception as e:
        raise Exception(f"Error while initialising values for plotting: {e}")

    # Plot data and principal components
    # decision against match-case statement, might not work in older python versions (< 3.10)
    try:
        if plot_type == '2D':
            tops, scaled_arrows = compute_arrows(pca, score, n_arrows, scales, 2)

            fig = plt.figure()
            ax = fig.add_subplot(111)

            if c:
                scatter = ax.scatter(xs * scalex, ys * scaley,
                                     c=c, cmap='tab10', alpha=0.5, zorder=0)
                legend_handles, legend_labels = scatter.legend_elements(prop="colors")
                ax.legend(legend_handles, legend_labels, title="Cluster")

            else:
                ax.scatter(xs * scalex, ys * scaley, alpha=0.5, zorder=0)

            for i, arrow in zip(tops, scaled_arrows):
                ax.quiver(0, 0, *arrow,
                          color='gray',
                          zorder=3,
                          angles='xy', scale_units='xy',
                          scale=1)
                ax.text(*(arrow * 1.15), features[i], ha='center', va='center')

            for i, axis in enumerate('xy'):
                getattr(ax, f'set_{axis}label')(f'PC{i + 1} ({pvars[i]:.2f}%)')

            # make up for plot
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            plt.grid()
            plt.title('2D Biplot')

        elif plot_type == '3D':
            tops, scaled_arrows = compute_arrows(pca, score, n_arrows, scales, 3)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for i, arrow in zip(tops, scaled_arrows):
                ax.quiver(0, 0, 0, *arrow,
                          color='gray',
                          zorder=3)
                ax.text(*(arrow * 1.15), features[i], ha='center', va='center')

            # plot points
            if c:
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
            warnings.warn(f"Warning: Unknown plot type '{plot_type}'")

        # save figure
        if savefig:
            check_path(savefig)
            plt.savefig(savefig, bbox_inches='tight', dpi=300)
        plt.show()

    except Exception as e:
        raise Exception(f"Error while plotting: {e}")


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
                         "If neither is provided, n_features defaults to 6.")

    # set n_features to the amount of given features by list
    if features:
        n_features = len(features)

    # get n_features most influencial features from pca
    else:
        # circular import problem
        from .computation import compute_pca
        # repetition of features or labels in feature columns...
        features = compute_pca(df.drop(label, axis=1), plot_type=None,
                               comp=n_features).drop(['Expl_var', 'Expl_var_ratio']).idxmax().reset_index()

        # nicer solution?
        add_features = 1
        while features[0].nunique() != n_features:
            features = compute_pca(df.drop(label, axis=1), plot_type=None,
                                   comp=n_features + add_features).drop(
                ['Expl_var', 'Expl_var_ratio']).idxmax().reset_index()
            add_features += 1
        features = features[0].unique()

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
        check_path(savefig)
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
    plt.show()
    return


# Violin plot for each cluster to compare their characteristics
def plot_violin(df, label='label', **kwargs):
    pass
