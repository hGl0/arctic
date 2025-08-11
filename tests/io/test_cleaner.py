import logging
import pandas as pd
import numpy as np
import pytest

from vortexclust.io.cleaner import *

logger = logging.getLogger(__name__)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1,2,3],
        'B': [10, 20, 30],
        '   foo' : ['  foo ', ' bar', 'baz  '],
        'num' : ['1', '2', '3'],
        'date': ['2020-01-01', '2021-05-10', '2022-12-31']
    })

def test_no_white_space(sample_df):
    df = no_white_space(sample_df)

    assert list(df.columns) == ['A', 'B', 'foo', 'num', 'date']
    assert df['foo'].tolist() == ['foo', 'bar', 'baz']
    assert df['num'].tolist() == ['1', '2', '3']

    df = pd.DataFrame()
    result = no_white_space(df)
    assert result.empty
    assert result.columns.tolist() == []

def test_no_white_space_invalid():
    with pytest.raises(TypeError, match="df must be a DataFrame"):
        no_white_space("Not a dataframe")


def test_to_date(sample_df):
    to_date(sample_df, 'date')

    assert pd.api.types.is_datetime64_any_dtype(sample_df.date)
    df_mixed = pd.DataFrame({'date': ['2020-01-01', 'May 10, 2021', '31/12/2022']})
    to_date(df_mixed, 'date')
    assert pd.api.types.is_datetime64_any_dtype(df_mixed.date)

def test_to_date_invalid():
    df = pd.DataFrame({'date': ['2020-01-01', 'invalid-date']})

    with pytest.raises(ValueError):
        to_date(df, 'date', format='%Y-%m-%d')