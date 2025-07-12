import logging

import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import pytest

from arctic.analysis import *

logger = logging.getLogger(__name__)

@pytest.fixture
def sample_data():
    X, _ = make_blobs(n_samples=200,
                      n_features=5,
                      centers=3,
                      cluster_std=1)
    return X

# tests regarding gap statistic
def test_gap_statistic(sample_data):
    k_max = 10
    result = gap_statistic(sample_data, k_max)
    for k in range(1, k_max):
        if result[k][0] >= result[k + 1][0] - result[k + 1][1]:
            kopt = k+1
            logger.info(f"Gap statistic optimal k: {kopt} \n"
                        f"Gap statistic:\n {result}")
            break
    assert isinstance(result, np.ndarray)
    assert result.shape == (k_max, 2)
    assert not np.isnan(result).any()
    assert (result[:, 1]>0).all()

def test_gap_statistic_invalid(sample_data):
    with pytest.raises(ValueError, match="Maximum number of clusters"):
        gap_statistic(sample_data, k_max=0)
    with pytest.raises(ValueError, match="Number of reference data sets"):
        gap_statistic(sample_data, k_max=5, n_replicates=0)

# tests regarding elbow method
def test_elbow_method(sample_data):
    k_max = 10
    result = elbow_method(sample_data, k_max)
    distortions = result[0]
    inertias = result[1]
    logger.info(f"Elbow method:\n {result}")
    assert isinstance(result, tuple)
    assert isinstance(distortions, list)
    assert isinstance(inertias, list)
    assert all(isinstance(x, (int, float)) for x in distortions)
    assert all(isinstance(x, (int, float)) for x in inertias)
    assert len(result) == 2
    assert len(distortions) == k_max
    assert len(inertias) == k_max
    assert not np.isnan(result).any()

def test_elbow_method_invalid(sample_data):
    with pytest.raises(ValueError, match="Maximum number of clusters"):
        elbow_method(sample_data, k_max=0)
    with pytest.raises(TypeError, match="X must be"):
        elbow_method('not_a_ndarray', k_max=5)

# tests regarding silhouette method
def test_silhouette_method(sample_data):
    k_max = 10
    result = silhouette_method(sample_data, k_max)
    logger.info(f"Silhouette method optimal k: {pd.DataFrame(result).idxmax()[0]+2}")
    logger.info(f"Silhouette method:\n {result}")
    assert isinstance(result, list)
    assert len(result) == k_max-1
    assert all(isinstance(x, (int, float)) for x in result)
    assert not np.isnan(result).any()

def test_silhouette_method_invalid_kmax(sample_data):
    with pytest.raises(ValueError, match="k_max must be at least"):
        silhouette_method(sample_data, k_max=0)



