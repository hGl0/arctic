import logging
import pytest

from vortexclust.visualization.correlation import *

logger = logging.getLogger(__name__)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1,2,3,4],
        'B': [10, 20, 30, 40],
        'C': [9, 8, 7, 6],
    })

def test_plot_correlation(sample_df, tmp_path):
    # correct input
    result = plot_correlation(sample_df)
    assert isinstance(result, pd.io.formats.style.Styler)

    # empty input
    empty = pd.DataFrame()
    with pytest.warns(UserWarning, match="The provided DataFrame is empty."):
        result = plot_correlation(empty)
        assert result is None

    # savefig and savecsv
    savefig_path = tmp_path / 'correlation.png'
    savecsv = tmp_path / 'correlation_matrix.csv'

    result = plot_correlation(sample_df, savefig=str(savefig_path), savecsv=str(savecsv))
    assert savefig_path.exists()
    assert savecsv.exists()
    assert isinstance(result, pd.io.formats.style.Styler)


def test_plot_correlation_invalid(sample_df):
    with pytest.raises(TypeError):
        plot_correlation("Not a DataFrame")

    with pytest.warns(UserWarning, match="Invalid colormap"):
        result = plot_correlation(sample_df, cmap="Not_a_cmap")
        assert isinstance(result, pd.io.formats.style.Styler)

    invalid_path = "this/path/does/not/exist/fig.png"
    with pytest.raises(FileNotFoundError):
        plot_correlation(sample_df, savefig=invalid_path)

