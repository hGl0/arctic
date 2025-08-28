import logging

import pandas as pd
import numpy as np
import pytest

from vortexclust.core.utils import *

logger = logging.getLogger(__name__)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1,2,3,4],
        'B': [10, 20, 30, 40],
    })

def test_norm_series_df(sample_df):
    result = norm_series_df(sample_df)
    assert isinstance(result, pd.DataFrame)

def test_norm_series_df_invalid(sample_df):
    with pytest.raises(TypeError):
        norm_series_df("not_a_serie")
    with pytest.raises(TypeError):
        norm_series_df(np.array([1, 2, 3]))

def test_validate_columns(sample_df):
    validate_columns(sample_df, ['A', 'B'])


def test_validate_columns_invalid(sample_df):
    # check if missing is detected
    with pytest.raises(KeyError):
        validate_columns(sample_df, ['A', 'B', 'C'])
    with pytest.raises(KeyError):
        validate_columns(pd.DataFrame(), ['A', 'B'])
    # dealing with series
    with pytest.raises(TypeError):
        test = pd.Series([1, 2, 3])
        validate_columns(test, ['A', 'B'])
    # dealing with other not dataframe types
    with pytest.raises(TypeError):
        validate_columns("not_a_df", 'A')
    # warning when empty required columns
    with pytest.warns(UserWarning, match = "At least one required column"):
        validate_columns(sample_df, [])