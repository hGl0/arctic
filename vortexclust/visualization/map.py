import math
from typing import Union, Tuple, Optional
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import animation

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

from vortexclust.analysis.aggregator import apply_aggregation
from vortexclust.analysis.geometry import compute_ellipse
from vortexclust.core.utils import norm_series_df, validate_columns
from vortexclust.io.paths import check_path

# To Do Doc string
def plot_split(ax, sample, filled, color='red'):
    # plot mother vortex
    x_final, y_final, _ = compute_ellipse(sample.area, sample.ar, sample.theta, sample.loncent, sample.latcent)
    ax.plot(x_final, y_final, linestyle='-.', color=color, label='Mother vortex')

    # plot daughter vortices
    for n in [1, 2]:
        if pd.notnull(sample.get(f'area{n}')):
            x, y, _ = compute_ellipse(sample[f'area{n}'], sample[f'ar{n}'], sample[f'theta{n}'],
                                      sample[f'loncent{n}'], sample[f'latcent{n}'])
            plot_ellipse(ax, x, y, sample[f'loncent{n}'], sample[f'latcent{n}'], filled=filled,
                         label=f"Split vortex {n}", color=color)

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
    :param split_col:
    :type split_col: str
    :param split:
    :type split: int
    :param figsize:
    :type figsize: Tuple[int]
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
                 label: str = 'Vortex ellipse',
                 center_color: str='blue') -> None:
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
    ax.scatter(loncent, latcent, color=center_color, marker='x', transform=ccrs.PlateCarree(), label='Center')
    ax.legend(loc='upper left')



def plot_polar_stereo(
        df: Union[pd.DataFrame, pd.Series],
        mode: str = 'single',  # modify mode on given arguments?
        **kwargs) -> None:
    r"""
    Plots polar stereographic projections of ellipses representing polar vortices.

    This function supports multiple visualization modes: single plot, aggregated plot, 
    animation, subplots, or overlay of multiple ellipses.

    :param df: DataFrame containing ellipse parameters (area, latcent, loncent, theta, ar)
    :type df: pd.DataFrame or pd.Series
    :param P: Pressure level in hPa, defaults to 10. Reserved for future use
    :param T: Temperature in degrees Celsius, defaults to -50. Reserved for future use.
    :param mode: Visualization mode, one of 'single', 'aggregate', 'animate', 'subplot', or 'overlay', defaults to 'single'
    :type mode: str, optional
    :param kwargs: Additional arguments:
        - `agg_func` (str or callable, optional): Aggregation function for 'aggregate' mode
        - `time_col` (str, optional): Column name containing time data, required for 'animate' mode
        - `max_subplots` (int, optional): Maximum number of subplots for 'subplot' mode, defaults to 4
        - `savefig` (str, optional): Path to save the figure, defaults to None
        - `filled` (bool, optional): Whether to fill ellipses or just draw outlines, defaults to False
        - `cmap` (str, optional): Colormap for ellipses, defaults to 'viridis'
        - `max_brightness` (float, optional): Maximum brightness for colormap, defaults to 0.7

    :raises KeyError: If 'time_col' is not provided for 'animate' mode
    :raises ValueError: If required columns are missing from the DataFrame

    :return: None
    """
    agg_func = kwargs.get('agg_func', None)
    time_col = kwargs.get('time_col', None)
    max_subplots = kwargs.get('max_subplots', 4)
    savefig = kwargs.get('savefig', None)
    filled = kwargs.get('filled', False)
    cmap = kwargs.get('cmap', 'viridis')
    max_brightness = kwargs.get('max_brightness', 0.7)
    split = kwargs.get('split', 1) # does it make sense?
    split_col = kwargs.get('split_col', 'form')
    figsize = kwargs.get('figsize', (10, 10))

    # for future use
    """lon = np.linspace(0, 360, 361)
    lat = np.linspace(90, 40, 71)  # From the North Pole to mid-latitudes
    lon2d, lat2d = np.meshgrid(lon, lat)

    # compute geopotential height
    T = T + ZERO_DEG  # convert T to Kelvin
    z = R * T / g * np.log(P0 / P)
    gph = np.full_like(lon2d, z)
    """

    # deal with differences between pd.Series and pd.DataFrame
    df = norm_series_df(df)
    validate_columns(df, ['area', 'latcent', 'loncent', 'theta', 'ar'])

    if mode == "aggregate":
        df = apply_aggregation(df, agg_func=agg_func)
        plot_polar_stereo(df, savefig=savefig, figsize=figsize)

    elif mode == "animate":
        if time_col is None:
            raise KeyError("time_col required for animation mode")
        out = savefig
        if out:
            p = Path(out)
            if p.suffix.lower() not in {".gif"}:
                # coerce to GIF, or warn
                out = str(p.with_suffix(".gif"))
        validate_columns(df, [time_col])
        # To Do: adjust for split event
        create_animation(df, time_col, filled, savegif=out, figsize=figsize)

    elif mode == "subplot":
        count = min(len(df), max_subplots)
        ncols = math.ceil(math.sqrt(count))
        nrows = math.ceil(count / ncols)

        # Sort and check time column
        if time_col and time_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(by=time_col).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
            time_col = None

        groups = [df.loc[idx] for idx in np.array_split(df.index, count)]
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                                subplot_kw={'projection': ccrs.NorthPolarStereo()})
        axs = axs.flatten()

        for i, (ax, group) in enumerate(zip(axs, groups)):
            ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.NaturalEarthFeature('physical', 'coastline', '50m',
                                                        edgecolor='black', facecolor='none'), linewidth=0.5)

            if len(group) > 10:
                print(f"Warning: Subplot {i + 1} has {len(group)} ellipses. This may be hard to distinguish.")

            # Color setup
            if time_col:
                colors = cm.viridis(np.linspace(0, 1, len(group)))
                labels = group[time_col].dt.strftime("%Y-%m-%d").tolist()
                title = f"{labels[0]} to {labels[-1]}"
            else:
                colors = cm.tab10(np.linspace(0, 1, len(group)))
                labels = [f"PV {j + 1}" for j in range(len(group))]
                title = f"Subplot {i + 1}"

            for j, (_, row) in enumerate(group.iterrows()):
                if row.get(split_col) == split:
                    plot_split(ax, row, filled, color=colors[j])
                else:
                    x, y, _ = compute_ellipse(row.area, row.ar, row.theta, row.loncent, row.latcent)
                    plot_ellipse(ax, x, y, row.loncent, row.latcent, filled=filled,
                                    color=colors[j], label=labels[j])

            # Add gridlines: 8 longitude lines every 45°, latitude every 10°
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                              draw_labels=True,
                              color='gray',
                              linestyle='--',
                              linewidth=0.8)

            gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 45))
            ax.set_title(title)
            ax.legend(loc='lower left', fontsize='small')

            # Hide unused subplots
        for j in range(len(groups), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        if savefig:
            check_path(savefig)
            plt.savefig(savefig)
        plt.show()

    elif mode == "overlay":
        original_cmap = plt.colormaps.get_cmap(cmap)
        trimmed_colors = original_cmap(np.linspace(0, max_brightness, 256))
        muted_cmap = ListedColormap(trimmed_colors)

        fig, ax = create_polar_ax(figsize=figsize)

        # create color gradient if time column is available
        if time_col is not None:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(by=time_col).reset_index(drop=True)
            c = muted_cmap(np.linspace(0, 1, len(df)))
        else:
            c = ['tab:blue'] * len(df)

        # Plot each sample with appropriate color
        for i, row in df.iterrows():
            if row.get(split_col) == split:
                plot_split(ax, row, filled, color=c[i])
            else:
                x, y, _ = compute_ellipse(row.area, row.ar, row.theta, row.loncent, row.latcent)
                label = 'Vortex ellipse' if i == 0 else None

                if filled:
                    ax.fill(x, y, color=c[i], alpha=0.5, label=label) if filled else ax.plot(x, y, color=c[i],
                                                                                              label=label)
                else:
                    ax.plot(x, y, color=c[i], label=label)

                center_label = 'Center' if i == 0 else None
                ax.scatter(row.loncent, row.latcent, color=c[i], marker='x',
                       transform=ccrs.PlateCarree(), label=center_label, zorder=5)
        if time_col:
            norm = mcolors.Normalize(vmin=0, vmax=len(df) - 1)
            sm = cm.ScalarMappable(cmap=muted_cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal')
            cbar.set_label(f'Time progression ({df[time_col].iloc[0].date()} → {df[time_col].iloc[-1].date()})')

        # Add gridlines: 8 longitude lines every 45°, latitude every 10°
        gl = ax.gridlines(crs=ccrs.PlateCarree(),
                          draw_labels=True,
                          color='gray',
                          linestyle='--',
                          linewidth=0.8)

        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 45))

        # One legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper left')

        fig.autofmt_xdate(rotation=45)
        if savefig:
            check_path(savefig)
            plt.savefig(savefig)
        plt.show()

    else:
        for _, sample in df.iterrows():
            # extract parameters for ellipse computation
            area = sample.area  # km², example area
            latcent = sample.latcent  # Latitude of vortex center
            loncent = sample.loncent  # Longitude of vortex center
            theta = sample.theta  # Angle of the major axis in degrees
            ar = sample.ar  # Aspect ratio (major / minor axis)

            x_final, y_final, _ = compute_ellipse(area, ar, theta, loncent, latcent)

            fig, ax = create_polar_ax(figsize=figsize)
            color = plt.get_cmap(cmap)(0.0)
            if time_col is not None:
                t=pd.to_datetime(sample[time_col])
                ax.set_title(t.strftime("%d-%m-%Y %H:%M"))

            if sample.get(split_col) == split:
                plot_split(ax, sample, filled, color=color)
            else:
                plot_ellipse(ax, x_final, y_final, loncent, latcent, color = color, filled=filled)

            # Add gridlines: 8 longitude lines every 45°, latitude every 10°
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                              draw_labels=True,
                              color='gray',
                              linestyle='--',
                              linewidth=1)
            gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 45))

            # Add a legend
            ax.legend(loc='upper left')
            fig.autofmt_xdate(rotation=45)
            fig.tight_layout()

            if savefig:
                check_path(savefig)
                plt.savefig(savefig)
            plt.show()

            # Plot data as filled contours, show changing gph as shades of grey
            # include color bar if so
            # contour = ax.contourf(lon2d, lat2d, gph, levels=20, transform=ccrs.PlateCarree(), cmap='Greys')

            # # Add contour lines for gph or something else
            # ax.contour(lon2d, lat2d, gph, levels=10, colors='black', linewidths=1, transform=ccrs.PlateCarree())

            # Add a colorbar
            # cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            # cbar.set_label('Geopotential Height (gpm)')