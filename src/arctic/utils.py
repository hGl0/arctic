import os.path
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import List, Tuple, Optional
import pandas as pd


def check_path(save_path: str) -> None:
    r"""
    Ensures that a given path is valid and exists. Raises an error otherwise.

    :param save_path: The file path where data will be saved.

    :raises TypeError: If 'save_path' is not a string.
    :raises FileNotFoundError: If the directory does not exist.

    :return: None
    """
    if not isinstance(save_path, str):
        raise TypeError(f"Expected a string (existing file path), but got {type(save_path).__name__}.")

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        raise FileNotFoundError(f"Path '{save_path}' does not exist.\n"
                                f"Please create it before saving, or give a valid path.")


def check_unfitted_model(model) -> None:
    r"""
    Checks if a clustering model is unfitted and has required attributes.

    :param model: A clustering model instance to be validated.
    :type model: object

    :raises TypeError: If the model is already fitted or doesn't implement a callable fit_predict method.
    :raises ValueError: If the model doesn't support the 'n_clusters' parameter.

    :return: None
    """
    # check if model is fitted, does not support n_clusters or fit_predict
    if hasattr(model, 'labels_') or hasattr(model, 'fit_predict_called'):
        raise TypeError("Passed model appears to be fitted. Please provide an unfitted model instance.")
    if not hasattr(model, 'fit_predict') or not callable(getattr(model, 'fit_predict')):
        raise TypeError("Provided model must implement a callable `fit_predict(X)` method.")
    if 'n_clusters' not in model.get_params():
        raise ValueError("Provided model must support the 'n_clusters' parameter via set_params().")


def compute_arrows(pca,
                   score: np.ndarray,
                   n_arrows: int,
                   scales: List[float],
                   n: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Computes and scales the top `n_arrows` vectors for a PCA biplot.

    :param pca: Fitted PCA object containing the principal components.
    :param score: A NumPy array of PCA scores.
    :param n_arrows: The number of top features (arrows) to select.
    :param scales: A list of scaling factors for each principal component.
    :param n: The number of principal components to consider.

    :raises ValueError: If `pca` does not contain 'components_' or
        if `score` has fewer than `n_components` columns.

    :return: A tuple containing:
        - An array of indices corresponding to the selected features.
        - A NumPy array of scaled arrows.
    """
    if not hasattr(pca, "components_"):
        raise ValueError("The provided PCA object must have 'components_' attribute.")
    if score.shape[1] < n:
        raise ValueError(f"Score matrix must have at least {n} columns")

    coeff = pca.components_[:n].T

    # find 'n_arrows' longest arrows
    tops = (coeff ** 2).sum(axis=1).argsort()[-n_arrows:]
    arrows = coeff[tops, :n]

    # Scale arrows
    norm = np.sqrt((arrows ** 2)).sum(axis=0)
    norm[norm == 0] = 1  # avoid division by zero
    arrows /= norm
    arrows *= np.abs(score[:, :n]).max(axis=0)
    scaled_arrows = arrows * np.array(scales[:n])

    return tops, scaled_arrows


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

    # circular import workaround
    from .computation import compute_pca

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



def norm_series_df(df):
    r"""
    Normalizes a pandas Series or DataFrame by ensuring it's a DataFrame with reset index.

    :param df: Input data to normalize
    :type df: pd.Series or pd.DataFrame

    :raises TypeError: If input is neither a pandas Series nor DataFrame

    :return: A pandas DataFrame with reset index
    :rtype: pd.DataFrame
    """
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("Expected pd.Series or pd.DataFrame")
    return df.reset_index(drop=True)


def validate_columns(df, required_cols):
    r"""
    Validates that a DataFrame contains all required columns.

    :param df: The DataFrame to validate
    :type df: pd.DataFrame
    :param required_cols: List of column names that must be present in the DataFrame
    :type required_cols: list

    :raises ValueError: If any required columns are missing from the DataFrame

    :return: None
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def create_polar_ax():
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


def plot_ellipse(ax, x, y, loncent, latcent,
                 filled=True,
                 color='red',
                 label = 'Vortex ellipse'):
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


def create_animation(df, time_col, filled, savegif=None):
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
    import matplotlib.animation as animation
    from matplotlib import pyplot as plt
    from src.arctic.computation import compute_ellipse

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



def apply_aggregation(df, agg_func):
    r"""
    Applies an aggregation function to a DataFrame.

    :param df: The DataFrame to aggregate
    :type df: pd.DataFrame
    :param agg_func: Aggregation function to apply, can be a string ('mean', 'median', 'sum') or a callable
    :type agg_func: str or callable

    :raises ValueError: If agg_func is not a valid aggregation function

    :return: The aggregated DataFrame or Series
    :rtype: pd.DataFrame or pd.Series
    """
    if callable(agg_func):
        return df.apply(agg_func)
    elif isinstance(agg_func, str) and hasattr(df, agg_func):
        return getattr(df, agg_func)()
    else:
        raise ValueError("Invalid 'agg_func'. Use 'mean', 'median', 'sum', or a callable function.")
