import logging
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from vortexclust.visualization.pca import plot_pca
from vortexclust.analysis import compute_pca

logger = logging.getLogger(__name__)


@pytest.fixture

def sample_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    pca = PCA(n_components=3)
    x_reduced = pca.fit_transform(X)
    labels = pd.Series(np.random.randint(0, 3, size=100))
    features = [f"feature{i}" for i in range(X.shape[1])]
    return pca, x_reduced, features, labels

def test_plot_pca(sample_data, tmp_path):
    pca, x_reduced, features, labels = sample_data
    fig_path = tmp_path / "pca.png"

    # test 2D plot
    with patch("matplotlib.pyplot.show"):
        plot_pca(pca, x_reduced, features, labels, plot_type='2D', n_arrows=3)
        plt.close()
        # test 3D plot
        plot_pca(pca, x_reduced, features, labels, plot_type='3D', n_arrows=3)
        plt.close()
        # test savefig
        plot_pca(pca, x_reduced, features, labels, savefig=str(fig_path))
        plt.close()
        assert fig_path.exists()

    # test fallback features
    with patch("matplotlib.pyplot.show"):
        plot_pca(pca, x_reduced, labels=labels)
        plt.close()
        # test fallback labels
        plot_pca(pca, x_reduced, features=features)
        plt.close()
        # test minimal arguments
        plot_pca(pca, x_reduced)
        plt.close()
    with patch("matplotlib.pyplot.show"):
        # test too many arrows
        plot_pca(pca, x_reduced, n_arrows=26)
        plt.close()


def test_plot_pca_invalid(sample_data):
    pca, x_reduced, features, labels = sample_data
    with patch("matplotlib.pyplot.show"):
        with pytest.warns(UserWarning, match="Warning: Unknown plot"):
            plot_pca(pca, x_reduced, features, labels, plot_type='invalid')

def test_integration(tmp_path):
    X = pd.DataFrame(np.random.normal(size=(100, 5)), columns=[f"feat{i}" for i in range(5)])
    x_new, components, pca = compute_pca(X, n_comp=3)

    fig_path = tmp_path / "pca.png"
    with patch("matplotlib.pyplot.show"):
        plot_pca(pca, x_new, X.columns.tolist(), savefig=str(fig_path))
        plt.close()

    assert fig_path.exists()


