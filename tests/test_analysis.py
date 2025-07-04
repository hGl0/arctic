import pytest
import numpy as np

from arctic.analysis import *

# exposed funcitions
# from clustering
def test_gap_statistic():
    result = gap_statistic()
    assert isinstance(result, np.ndarray)

def test_elbow_method():
    result = elbow_method()
    assert isinstance(result, np.ndarray)

def test_silhouette_method():
    result = silhouette_method()
    assert isinstance(result, np.ndarray)

# from decomposition
def test_compute_pca():
    result = compute_pca()
    assert isinstance(result, np.ndarray)

def test_compute_eeof():
    result = compute_eeof()
    assert isinstance(result, np.ndarray)

# from metrics
def test_within_cluster_dispersion():
    result = within_cluster_dispersion()
    assert isinstance(result, np.ndarray)

# not exposed functions
from arctic.analysis.aggregator import apply_aggregation
def test_apply_aggregation():
    result = apply_aggregation()
    assert isinstance(result, np.ndarray)

from arctic.analysis.geometry import compute_ellipse
def test_compute_ellipse():
    result = compute_ellipse()
    assert isinstance(result, np.ndarray)

from arctic.analysis.sampling import generate_reference_HPPP
def test_generate_reference_HPPP():
    result = generate_reference_HPPP()
    assert isinstance(result, np.ndarray)

