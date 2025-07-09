import logging
import pytest
import numpy as np

from arctic.analysis import *

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("X, labels",
                         [(np.array([[0,0],[1,1],[2,2]]), np.array([0,0,1])),
                          (np.random.rand(10,3), np.array([0]*5+[1]*5))])

def test_within_cluster_dispersion(X, labels):
    result = within_cluster_dispersion(X, labels)
    assert isinstance(result, float)
    assert result >= 0

    X = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
    labels = np.array([0, 0, 1, 1])
    result = within_cluster_dispersion(X, labels)
    assert np.isclose(result, 0.0)

def test_within_cluster_dispersion_invalid():
    X = np.array([[0, 0], [1, 1], [2, 2]])
    labels = np.array([0, 0])  # Mismatched length
    with pytest.raises(ValueError, match="Mismatched dimensions"):
        within_cluster_dispersion(X, labels)
    with pytest.raises(TypeError):
        within_cluster_dispersion("not_an_array", labels)
    with pytest.raises(TypeError):
        within_cluster_dispersion(X, "not_an_array")

