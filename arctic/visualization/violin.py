import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from arctic.io.paths import check_path
from arctic.visualization.utils import feature_consistence


def plot_violin(df: pd.DataFrame, label: str = 'label', **kwargs) -> None:
    r"""
    Plots violin plots to compare feature distributions across clusters.

    :param df: (pd.DataFrame) DataFrame containing numerical features and cluster labels.
    :param label: (str) Column name used as the group label. Default is 'label'.
    :param kwargs: Additional arguments:
        - `features` (list, optional): List of features to plot. Default is None.
        - `n_features` (int, optional): Number of most important features to include. Default is 6.
        - `scaler` (object, optional): Scaler to normalize data. Default is MinMaxScaler.
        - `spacing` (float, optional): Spacing between violin plots. Default is 0.3.
        - `savefig` (str, optional): File path to save the figure. Default is None.

    :raises ValueError: If `n_features` is greater than available features.
    :raises FileNotFoundError: If `savefig` directory does not exist.

    :return: None
    """
    features = kwargs.get('features', None)  # List of features to plot
    n_features = kwargs.get('n_features', 6)  # Use the 6 most important features
    scaler = kwargs.get('scaler', MinMaxScaler)
    spacing = kwargs.get('spacing', 0.3)
    savefig = kwargs.get('savefig', None)  # location to save figure

    if label not in df.columns:
        raise KeyError(
            "Please give a valid label or ensure that your Pandas DataFrame contains a column named 'label'.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.drop('label', axis=1).dtypes):
        raise TypeError("Ensure your dataframe has only numeric types (except the label column).")

    n_features, features = feature_consistence(df.drop('label', axis=1), n_features, features)

    # Scale features for nicer look
    try:
        if isinstance(scaler, type):
            scaler = scaler()

            tmp = df.drop(label, axis=1)
            tmp = pd.DataFrame(scaler.fit_transform(tmp),
                               columns=tmp.columns, index=tmp.index)

            tmp[label] = df[label]
            df = tmp
    except TypeError as e:
        raise TypeError('Ensure your dataframe has only numeric types.')
    except Exception as e:
        raise Exception(f"Error while scaling: {e}")

    n_cluster = df[label].nunique()
    clusters = np.sort(df[label].unique())
    base_pos = [(n_cluster + 1) * spacing * i + 1 for i in range(n_features)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title('Violin plot')

    for i, l in enumerate(clusters):
        data = df[df[label] == l][features]

        pos = list(map(lambda x: np.round(x + i * spacing, 2), base_pos))
        ax.violinplot(data,
                      positions=pos,
                      showmeans=True, showextrema=False, widths=spacing)
        # generate labels with empty scatter, improve?
        ax.scatter([], [], label=l)

    plt.legend(title='Cluster')

    # set style for the axes
    offset = np.round(spacing * (n_cluster - 1), 2)

    ticks_pos = [x + offset / 2 for x in base_pos]
    ax.set_xticks(ticks_pos, labels=features)
    ax.set_xlim(base_pos[0] - spacing, base_pos[-1] + n_cluster * spacing)

    ax.set_xlabel('Feature')
    ax.set_ylabel(f'Scaled values ({scaler})')
    plt.subplots_adjust(bottom=0.15, wspace=0.05)
    if savefig:
        check_path(savefig)
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
    plt.show()
    return
