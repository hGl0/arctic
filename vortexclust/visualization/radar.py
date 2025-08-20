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

    if not label in df.columns:
        raise KeyError("Please give a valid label or ensure that your Pandas DataFrame contains a column named 'label'.")

    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.drop(label, axis=1).dtypes):
        raise TypeError("Ensure your dataframe has only numeric types (except the label column).")

    # set n_features to the amount of given features by list
    n_features, features = feature_consistence(df.drop(label, axis=1),
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

        for fi, f in enumerate(features):
            angle = angles[fi]

            # --- min and max across groups (original values) ---
            min_idx = val[f].idxmin()
            max_idx = val[f].idxmax()

            for g in [min_idx, max_idx]:
                value_scaled = val_scaled.loc[g, f]
                value_orig = val.loc[g, f]

                ax.text(angle, value_scaled + 0.06,
                        f"{value_orig:.2f}",
                        ha='center', va='center', fontsize=8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # ensure a fixed radial limit first (adjust if you use a different scale)
    rmin, rmax = ax.get_ylim()
    r_label = rmax * 1.2

    # add outward-facing labels at radius r_label
    for ang, txt in zip(angles[:-1], features):
        ang_deg = np.round(np.degrees(ang))

        rot = 0 # no rotation initially
        if ang_deg < 60:
            rot = ang_deg - 90
        if 60 <= ang_deg <= 120:
            rot = 0
        if 120 < ang_deg <= 180:
            rot = ang_deg - 90

        # rotate other direction in lower half
        if 180 < ang_deg <= 240:
            rot = ang_deg + 90
        if 240 <= ang_deg <= 300:
            rot = 0
        if 300 < ang_deg <= 360:
            rot = ang_deg + 90



        ax.text(
            ang, r_label, txt,
            ha="center", va="center",
            rotation=rot, rotation_mode="anchor",
            clip_on=False,  # don’t clip outside the axes
        )

    plt.legend(title='Cluster')
    plt.title(f"Radar Chart", y=1.05)

    if savefig:
        check_path(savefig)
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
    plt.show()
    return