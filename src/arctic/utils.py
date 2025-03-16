import os.path
import numpy as np

def check_path(save_path):
    """Ensures that a given path is valid, i.e. a string. Raises an error otherwise."""
    if not isinstance(save_path, str):
        raise TypeError(f"Expected 'savefig' to be string (existing file path), but got {type(save_path).__name__}.")

    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        raise FileNotFoundError(f"Path '{save_path}' does not exist.\n"
                                f"Please create it before saving, or give a valid path.")

def compute_arrows(pca, score, n_arrows, scalers, N):
    """Compute and scale the top n_arrows vectors for a PCA biplot"""
    coeff = pca.components_[:N].T

    # 1. approach: find n_arrows longest arrows
    tops = (coeff ** 2).sum(axis=1).argsort()[-n_arrows:]
    arrows = coeff[tops, :N]

    # 2. approach: find n_arrows features that drive most variance in the visible pcs
    # tops = (loadings*pvars).sum(axis=1).argsort()[-n_arrows:]
    # arrows = loadings[tops]

    # Scale arrows
    arrows /= np.sqrt((arrows ** 2)).sum(axis=0)
    arrows *= np.abs(score[:, :N]).max(axis=0)
    scaled_arrows = arrows * np.array(scalers[:N])

    return tops, scaled_arrows
