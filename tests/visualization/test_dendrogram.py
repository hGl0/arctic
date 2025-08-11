import logging
import pytest
from types import SimpleNamespace
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np

from vortexclust.visualization.dendrogram import plot_dendrogram

logger = logging.getLogger(__name__)

@pytest.fixture
def mock_model():
    return SimpleNamespace(
        children_=np.array([[0, 1], [2, 3]]),
        distances_=np.array([1.0, 2.0]),
        labels_=np.array([0, 1, 2, 3])
    )
def test_plot_dendrogram(mock_model, tmp_path):
    save_path = tmp_path / "dendrogram.png"

    with patch("matplotlib.pyplot.show"):
        # Should not raise
        plot_dendrogram(mock_model)
        plot_dendrogram(mock_model, savefig=str(save_path))
        plt.close()
        plt.close()

    assert save_path.exists()

def test_plot_dendrogram_invalid(mock_model, tmp_path):
    class IncompleteModel:
        children_ = np.array([[0, 1]])

    with patch("matplotlib.pyplot.show"):
        with pytest.raises(AttributeError, match="Model must have 'children_', 'distances_'"):
            plot_dendrogram(IncompleteModel())

    invalid_path = "this/path/does/not/exist/fig.png"
    with patch("matplotlib.pyplot.show"):
        with pytest.raises(FileNotFoundError):
            plot_dendrogram(mock_model, savefig=str(invalid_path))
