from typing import List, Tuple, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import animation

from arctic.analysis.decomposition import compute_pca
from arctic.analysis.geometry import compute_ellipse


def feature_consistence(df: pd.DataFrame,
                        n_features: Optional[int] = None,
                        features: Optional[List[str]] = [],
                        ) -> Tuple[int, List[str]]:
    r"""
    Determines the most influential PCA features or uses a given feature list.

    :param df: A pandas DataFrame containing the dataset.
    :param n_features: The number of influential features to select.
    :param features: A list of predefined features (optional).

    :raises ValueError: If neither `n_features` nor `features` is provided.
    :raises Exception: If the loop exceeds the number of available features in `df`.

    :return: A tuple containing:
        - The number of selected features.
        - A list with the feature names
    """
    # at least one must be given
    if (features == []) and (n_features is None):
        raise ValueError("Either 'n_features' or 'features' must be provided.")

    # n_features has valid values
    if (not n_features is None and
            ((n_features == 0) or (not isinstance(n_features, int)))):
        raise ValueError("If 'n_features' is provided, it must be an integer greater than 0.")

    if features:
        return len(features), features

    if n_features == df.shape[1]:
        return n_features, df.columns

    # # if n_features > features: add important pca features until n_features is reached
    # if (n_features > len(features)) and (len(features) > 0):
    #     pca_features = (pd.DataFrame(compute_pca(df, plot_type=None,
    #                                 n_comp=n_features)[1])
    #                     .drop(['Expl_var', 'Expl_var_ratio'])
    #                     .idxmax()
    #                     .reset_index())
    #     unique_features = pca_features[0].unique()
    #
    #     for feat in unique_features:
    #         if feat not in features:
    #             features.append(feat)
    #         if len(features) == n_features:
    #             return n_features, features
    #
    #     # reached when n_features > len(features), but no features could be added
    #     # example: features = df.columns, n_features = len(df.columns)+1
    #     warnings.warn(UserWarning(f"Could not add further features to reach {n_features} features."
    #                               f"Returning {len(features)} features."))
    #     return len(features), features
    #
    # # if n_features < len(features): select most important features from features
    # if (n_features < len(features)) and (len(features) > 0):
    #     pca_features = (pd.DataFrame(compute_pca(df[features], plot_type=None,
    #                                              n_comp=n_features)[1])
    #                     .drop(['Expl_var', 'Expl_var_ratio'])
    #                     .idxmax()
    #                     .reset_index())
    #     unique_features = pca_features[0].unique()
    #     return n_features, unique_features
    #

    add_features = 0
    max_attempts = df.shape[1]
    # based on only n_features
    while True:
        if add_features > max_attempts:
            raise Exception(f"Could not select exactly {n_features} unique features from PCA:\n"
                            f"Tried increasing but reached dataset limit.")

        pca_features = (pd.DataFrame(compute_pca(df, plot_type=None,
                                    n_comp=n_features + add_features)[1])
                        .drop(['Expl_var', 'Expl_var_ratio'])
                        .idxmax()
                        .reset_index())

        unique_features = pca_features[0].unique()

        if len(unique_features) == n_features:
            return n_features, list(unique_features)
        add_features += 1
    # n_features > features




def create_animation(df: pd.DataFrame, time_col: str, filled: bool,
                     savegif: Optional[str] = None,
                     split_col: Optional[str] = 'form',
                     split: Optional[int] = 1,
                     figsize: Optional[Tuple[int]] = (10,10)) -> None:
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

    fig = plt.figure(figsize=figsize)

    def update(frame):
        fig.clf()  # clear the figure entirely

        fig.suptitle(str(df.iloc[frame][time_col]))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())

        # Add coastlines and gridlines
        ax.coastlines()

        row = df.iloc[frame]
        color = 'tab:blue'

        if row.get(split_col) == split:
            from arctic.visualization.map import plot_split
            plot_split(ax, row, filled, color)
        else:
            x, y, _ = compute_ellipse(row.area, row.ar, row.theta, row.loncent, row.latcent)
            plot_ellipse(ax, x, y, row.loncent, row.latcent, filled)

        # Add gridlines: 8 longitude lines every 45°, latitude every 10°
        gl = ax.gridlines(crs=ccrs.PlateCarree(),
                          draw_labels=True,
                          color='gray',
                          linestyle='--',
                          linewidth=1)

        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 45))


    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=1000, repeat=False, blit=False)

    if savegif:
        ani.save(savegif, dpi=300, writer=animation.PillowWriter(fps=1))
    else:
        plt.show()
    plt.close()


def create_polar_ax(figsize=(10,10)) -> Tuple[plt.Figure, plt.Axes]:
    r"""
    Creates a matplotlib figure and axis with North Polar Stereographic projection.

    The function sets up a figure with coastlines and appropriate extent for Arctic visualization.

    :return: A tuple containing:
        - The matplotlib figure object
        - The axis object with North Polar Stereographic projection
    :rtype: tuple(matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.NorthPolarStereo()})
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
    # Coordinate validation
    if not isinstance(x, (np.ndarray, list)) or not isinstance(y, (np.ndarray, list)):
        raise TypeError("x and y must be array-like (e.g., np.ndarray or list).")

    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape. Got {x.shape} and {y.shape}.")

    if len(x) == 0:
        raise ValueError("x and y are empty.")

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must not contain NaN or infinite values.")

    if filled:
        ax.fill(x, y, color=color, label=label)
    else:
        ax.plot(x, y, color=color, label=label)
    ax.scatter(loncent, latcent, color=color, marker='x', transform=ccrs.PlateCarree(), label='Center')
    ax.legend(loc='upper left')


