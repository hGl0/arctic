from .correlation import plot_correlation
from .dendrogram import plot_dendrogram
from .pca import plot_pca
from .radar import plot_radar
from .violin import plot_violin

__all__ = [
    "plot_dendrogram",
    "plot_correlation",
    "plot_pca",
    "plot_radar",
    "plot_violin",
    "plot_polar_stereo",
]

# Lazy import for the cartopy-dependent function
import importlib

def plot_polar_stereo(*args, **kwargs):
    try:
        mod = importlib.import_module(".map", __package__)
    except ImportError as e:
        raise ImportError(
            "plot_polar_stereo requires Cartopy. "
            "Install with: pip install vortexclust[maps]"
        ) from e
    return mod.plot_polar_stereo(*args, **kwargs)
