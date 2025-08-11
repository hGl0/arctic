import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from vortexclust.analysis.aggregator import apply_aggregation
from vortexclust.io.paths import check_path
from vortexclust.visualization.utils import feature_consistence


def plot_radar(df: pd.DataFrame, label: str = 'label', **kwargs) -> None:
    r"""
    Plots a radar chart to visualize feature importance or differences across clusters.

    :param df: (pd.DataFrame) DataFrame containing feature values and cluster labels.
    :param label: (str) Column name used as the group label. Default is 'label'.
    :param kwargs: Additional arguments:
        - `features` (list, optional): List of features to plot. Default is None (selects top features).
        - `n_features` (int, optional): Number of most important features to include. Default is 6.
        - `savefig` (str, optional): File path to save the figure. Default is None.
        - `agg_func` (str or callable, optional): Aggregation function (e.g., 'mean', 'median'). Default is 'mean'.
        - `scaler` (object, optional): Scaler to normalize data. Default is StandardScaler.

    :raises ValueError: If `n_features` is greater than available features.
    :raises FileNotFoundError: If `savefig` directory does not exist.

    :return: None
    """
    features = kwargs.get('features', None)  # List of features to plot
    n_features = kwargs.get('n_features', 6)  # Use the 6 most important features
    savefig = kwargs.get('savefig', None)  # location to save figur
    agg_func = kwargs.get('agg_func', 'mean')  # aggregation function used with groupby
    scaler = kwargs.get('scaler', StandardScaler)

    if label not in df.columns:
        raise KeyError("Please give a valid label or ensure that your Pandas DataFrame contains a column named 'label'.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.drop('label', axis=1).dtypes):
        raise TypeError("Ensure your dataframe has only numeric types (except the label column).")

    # set n_features to the amount of given features by list
    n_features, features = feature_consistence(df.drop('label', axis=1),
                                               n_features=n_features,
                                               features=features)

    # group and aggregate dataframe
    # check for custom aggregation function
    grouped = df.groupby(label)
    val = apply_aggregation(grouped, agg_func)

    group = val.index.tolist()

    # Scale features for nicer look
    try:
        if isinstance(scaler, type):
            scaler = scaler()
            val_scaled = pd.DataFrame(
                scaler.fit_transform(val),
                columns=val.columns,
                index=val.index  # <- preserve group labels like 'A', 'B'
            )
        else:
            val_scaled = val
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