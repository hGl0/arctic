import logging
from unittest import mock
from unittest.mock import patch
import matplotlib.pyplot as plt
import pytest
import pandas as pd
import numpy as np

from vortexclust.visualization.utils import feature_consistence

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_df():
    np.random.seed(0)
    data = np.random.rand(20, 10)
    cols = [f'feat{i}' for i in range(data.shape[1])]
    return pd.DataFrame(data, columns=cols)

def test_feature_consistence(sample_df):
    # give number of features
    n, f = feature_consistence(df=sample_df, n_features=2)
    assert isinstance(f, list)
    assert n == len(f)

    # give list of features
    features = ['feat1', 'feat3']
    n, f = feature_consistence(df=sample_df, features=features)
    assert f == features
    assert n == len(f)

    sample_df['feat1'] = [1]*20
    sample_df['feat2'] = np.random.normal(size=20)
    # n_features > features: select features
    n, f = feature_consistence(df=sample_df, n_features=3, features=['feat1', 'feat2'])
    assert all(feat in f for feat in ['feat1', 'feat2'])
    assert n == 2
    assert n == len(f)

    # n_features > features: select features
    n, f = feature_consistence(df=sample_df, n_features=2, features=['feat1', 'feat2', 'feat3'])
    assert len(f) == n
    assert n == 3

def test_feature_consistence_invalid():
    # n_features=None and features=None
    with pytest.raises(ValueError, match="Either 'n_features' or 'features' must be provided."):
        feature_consistence(df=sample_df)

    with pytest.raises(ValueError, match="If 'n_features' is provided"):
        feature_consistence(df=sample_df, n_features=2.6)

    with pytest.raises(ValueError, match="If 'n_features' is provided"):
        feature_consistence(df=sample_df, n_features=2.6)