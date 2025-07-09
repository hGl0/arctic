import logging
import pandas as pd
import pytest
import numpy as np

from arctic.analysis.sampling import generate_reference_HPPP

logger = logging.getLogger(__name__)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1,2,3,4],
        'B': [10, 20, 30, 40],
    })

def test_generate_reference_HPPP(sample_df):
    result = generate_reference_HPPP(sample_df)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_df.shape

    logger.info(f"Minimum in sample:\n{sample_df.min(axis=0)}")
    logger.info(f"Minimum in HPPP:\n{result.min(axis=0)}")
    logger.info(f"Maximum in sample:\n{sample_df.max(axis=0)}")
    logger.info(f"Maximum in HPPP:\n{result.max(axis=0)}")

    assert (result.min(axis=0) >= sample_df.values.min(axis=0)).all()
    assert (result.max(axis=0) <= sample_df.max(axis=0).values).all()




