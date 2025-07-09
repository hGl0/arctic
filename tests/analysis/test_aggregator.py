import logging
import pandas as pd
import pytest

from arctic.analysis.aggregator import apply_aggregation

logger = logging.getLogger(__name__)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1,2,3,4],
        'B': [10, 20, 30, 40],
    })

def test_mean_aggregation(sample_df):
    result = apply_aggregation(sample_df, 'mean')
    expected = sample_df.mean()
    pd.testing.assert_series_equal(result, expected)

def test_sum_aggregation(sample_df):
    result = apply_aggregation(sample_df, 'sum')
    expected = sample_df.sum()
    pd.testing.assert_series_equal(result, expected)

def test_median_aggregation(sample_df):
    result = apply_aggregation(sample_df, 'median')
    expected = sample_df.median()
    pd.testing.assert_series_equal(result, expected)

def test_callable_aggregation(sample_df):
    result = apply_aggregation(sample_df, lambda x: x.max() - x.min())
    expected = sample_df.apply(lambda x: x.max() - x.min())
    pd.testing.assert_series_equal(result, expected)

def test_invalid_string_aggregation(sample_df):
    with pytest.raises(ValueError, match="Invalid 'agg_func'"):
        apply_aggregation(sample_df, 'not_a_method')

def test_invalid_type_aggregation(sample_df):
    with pytest.raises(ValueError):
        apply_aggregation(sample_df, 123)


