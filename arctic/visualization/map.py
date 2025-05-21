import pandas as pd
import numpy as np
import math
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


from arctic.io.paths import check_path
from arctic.analysis.geometry import compute_ellipse
from arctic.analysis.aggregator import apply_aggregation
from arctic.visualization.utils import create_polar_ax, plot_ellipse, create_animation
from arctic.core.utils import norm_series_df, validate_columns

from typing import Union


def plot_polar_stereo(
        df: Union[pd.DataFrame, pd.Series], P: Union[int, float] = 10, T: Union[int, float] = -50,
        mode: str = 'single',  # modify mode on given arguments?
        **kwargs) -> None:
    r"""
    Plots polar stereographic projections of ellipses representing polar vortices.

    This function supports multiple visualization modes: single plot, aggregated plot, 
    animation, subplots, or overlay of multiple ellipses.

    :param df: DataFrame containing ellipse parameters (area, latcent, loncent, theta, ar)
    :type df: pd.DataFrame or pd.Series
    :param P: Pressure level in hPa, defaults to 10
    :type P: int or float, optional
    :param T: Temperature in degrees Celsius, defaults to -50
    :type T: int or float, optional
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

    :raises AssertionError: If 'time_col' is not provided for 'animate' mode
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
        plot_polar_stereo(df)

    elif mode == "animate":
        assert time_col is not None, "time_col required for animation mode"
        create_animation(df, time_col, filled, savefig)

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

        groups = np.array_split(df, count)
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
                colors = cm.viridis(np.linspace(0.2, 0.9, len(group)))
                labels = group[time_col].dt.strftime("%Y-%m-%d").tolist()
                title = f"{labels[0]} to {labels[-1]}"
            else:
                colors = cm.tab10(np.linspace(0, 1, len(group)))
                labels = [f"PV {j + 1}" for j in range(len(group))]
                title = f"Subplot {i + 1}"

            for j, (_, row) in enumerate(group.iterrows()):
                x, y, _ = compute_ellipse(row.area, row.ar, row.theta, row.loncent, row.latcent)
                plot_ellipse(ax, x, y, row.loncent, row.latcent, filled=filled,
                                    color=colors[j], label=labels[j])

            ax.set_title(title)
            ax.legend(loc='lower left', fontsize='small')

            # Hide unused subplots
        for j in range(len(groups), len(axs)):
            fig.delaxes(axs[j])

        plt.tight_layout()
        plt.show()
    elif mode == "overlay":
        original_cmap = cm.get_cmap(cmap)
        trimmed_colors = original_cmap(np.linspace(0, max_brightness, 256))
        muted_cmap = ListedColormap(trimmed_colors)

        fig, ax = create_polar_ax()

        # create color gradient if time column is available
        if time_col is not None:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(by=time_col).reset_index(drop=True)
            c = muted_cmap(np.linspace(0, 1, len(df)))
        else:
            c = ['red'] * len(df)

        # Plot each sample with appropriate color
        for i, row in df.iterrows():
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
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05)
            cbar.set_label(f'Time progression ({df[time_col].iloc[0].date()} → {df[time_col].iloc[-1].date()})')

        # One legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper left')

        fig.autofmt_xdate(rotation=45)
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

            fig, ax = create_polar_ax()

            import matplotlib.ticker as mticker

            # Add gridlines: 8 longitude lines every 45°, latitude every 10°
            gl = ax.gridlines(crs=ccrs.PlateCarree(),
                              draw_labels=True,
                              color='gray',
                              linestyle='--',
                              linewidth=0.8)

            gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 45))
            # gl.ylocator = mticker.FixedLocator(np.arange(40, 91, 10))

            plot_ellipse(ax, x_final, y_final, loncent, latcent, filled=filled)
            # Add a legend
            ax.legend(loc='upper left')
            fig.autofmt_xdate(rotation=45)
            if savefig:
                check_path(savefig)
            plt.show()

            # Plot data as filled contours, show changing gph as shades of grey
            # include color bar if so
            # contour = ax.contourf(lon2d, lat2d, gph, levels=20, transform=ccrs.PlateCarree(), cmap='Greys')

            # # Add contour lines for gph or something else
            # ax.contour(lon2d, lat2d, gph, levels=10, colors='black', linewidths=1, transform=ccrs.PlateCarree())

            # Add a colorbar
            # cbar = plt.colorbar(contour, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            # cbar.set_label('Geopotential Height (gpm)')
