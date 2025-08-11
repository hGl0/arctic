import logging
import pandas as pd
import pytest

from vortexclust.io.loader import *

logger = logging.getLogger(__name__)

def test_read_data(tmp_path):
    # do not indent, else lines do not start with 'D'
    sample_content = """\
# This is a comment
D,col1,col2,col3
A,not included
D,1,2,3
X,wrong prefix
D,4,5,6
"""
    file_path = tmp_path / "data.csv"
    file_path.write_text(sample_content)
    df = read_data(str(file_path))

    logger.info(f"Read data from {file_path}\n{df}")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.iloc[0, 0] == "D"
    assert df.iloc[1, 0] == "D"
    assert df.iloc[0].tolist() == ["D", 1,2,3]
    assert df.iloc[1].tolist() == ["D", 4,5,6]


