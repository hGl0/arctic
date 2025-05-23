import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from typing import List, Tuple, Optional

from arctic.analysis.geometry import compute_ellipse
from arctic.analysis.decomposition import compute_pca




def feature_consistence(n_features: Optional[int],
                        features: Optional[List[str]],
                        df: pd.DataFrame) -> Tuple[int, List[str]]:
    r"""
    Determines the most influential PCA features or uses a given feature list.

    :param n_features: The number of influential features to select.
    :param features: A list of predefined features (optional).
    :param df: A pandas DataFrame containing the dataset.

    :raises ValueError: If neither `n_features` nor `features` is provided.
    :raises Exception: If the loop exceeds the number of available features in `df`.

    :return: A tuple containing:
        - The number of selected features.
        - A list with the feature names
    """

    if features:
        return len(features), features

    if n_features is None:
        raise ValueError("Either 'n_features' or 'features' must be provided.")

    if n_features == df.shape[1]:
        return n_features, df.columns

    add_features = 0
    max_attempts = df.shape[1]

    while True:
        if add_features > max_attempts:
            raise Exception(f"Could not select exactly {n_features} unique features from PCA:\n"
                            f"Tried increasing but reached dataset limit.")
        pca_features = (compute_pca(df, plot_type=None,
                                    comp=n_features + add_features)
                        .drop(['Expl_var', 'Expl_var_ratio'])
                        .idxmax()
                        .reset_index())

        unique_features = pca_features[0].unique()

        if len(unique_features) == n_features:
            return n_features, list(unique_features)
        add_features += 1

def create_animation(df: pd.DataFrame, time_col: str, filled: bool, savegif: Optional[str] = None) -> None:
    r"""
    Creates an animation of ellipses over time using a DataFrame of ellipse parameters.

    :param df: DataFrame containing ellipse parameters for each time step
    :type df: pd.DataFrame
    :param time_col: Name of the column in df that contains time information
    :type time_col: str
    :param filled: Whether to fill the ellipses or just draw outlines
    :type filled: bool
    :param savegif: Path to save the animation as a GIF file, if None the animation is displayed instead
    :type savegif: str, optional

    :return: None
    """

    df = df.sort_values(by=time_col)

    fig = plt.figure(figsize=(6, 6))

    def update(frame):
        fig.clf()  # clear the figure entirely

        fig.suptitle(str(df.iloc[frame][time_col]))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())

        # Add coastlines and gridlines
        ax.coastlines()
        ax.gridlines(draw_labels=False, linestyle='--', color='gray')

        row = df.iloc[frame]
        x, y, _ = compute_ellipse(row.area, row.ar, row.theta, row.loncent, row.latcent)
        plot_ellipse(ax, x, y, row.loncent, row.latcent, filled)

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=1000, repeat=False, blit=False)

    if savegif:
        ani.save(savegif, writer='pillow', fps=2)
    else:
        plt.show()


def create_polar_ax() -> Tuple[plt.Figure, plt.Axes]:
    r"""
    Creates a matplotlib figure and axis with North Polar Stereographic projection.

    The function sets up a figure with coastlines and appropriate extent for Arctic visualization.

    :return: A tuple containing:
        - The matplotlib figure object
        - The axis object with North Polar Stereographic projection
    :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': ccrs.NorthPolarStereo()})
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '50m',
                                                edgecolor='black', facecolor='none'), linewidth=0.5)
    ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    return fig, ax


def plot_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, loncent: float, latcent: float,
                 filled: bool = True,
                 color: str = 'red',
                 label: str = 'Vortex ellipse') -> None:
    r"""
    Plots an ellipse on a given axis, with options for filled or outline representation.

    :param ax: The matplotlib axis on which to plot
    :type ax: matplotlib.axes.Axes
    :param x: X-coordinates of the ellipse points
    :type x: array-like
    :param y: Y-coordinates of the ellipse points
    :type y: array-like
    :param loncent: Longitude of the ellipse center
    :type loncent: float
    :param latcent: Latitude of the ellipse center
    :type latcent: float
    :param filled: Whether to fill the ellipse or just draw the outline, defaults to True
    :type filled: bool, optional
    :param color: Color of the ellipse, defaults to 'red'
    :type color: str, optional
    :param label: Label for the ellipse in the legend, defaults to 'Vortex ellipse'
    :type label: str, optional

    :return: None
    """
    if filled:
        ax.fill(x, y, color=color, label=label)
    else:
        ax.plot(x, y, color=color, label=label)
    ax.scatter(loncent, latcent, color=color, marker='x', transform=ccrs.PlateCarree(), label='Center')
    ax.legend(loc='upper left')


