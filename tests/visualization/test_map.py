import logging
import pytest
import pandas as pd
from unittest.mock import patch
import matplotlib.pyplot as plt

from vortexclust.visualization.map import plot_polar_stereo

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'area': [1e6, 1e6],
        'latcent': [85, 80],
        'loncent': [0, 45],
        'theta': [30, 60],
        'ar': [1.5, 1.2]
    })


def test_plot_polar_stereo(sample_df, tmp_path):
    with patch("matplotlib.pyplot.show"):
        plot_polar_stereo(sample_df, mode='single')
        plt.close()

    # should look like single
    sample_df['time'] = pd.date_range('2020-01-01', periods=len(sample_df))
    with patch("matplotlib.pyplot.show"):
        plot_polar_stereo(sample_df, mode='overlay', time_col='time', max_subplots=4)
        plt.close()

    sample_df = pd.concat([sample_df] * 4, ignore_index=True)
    sample_df['time'] = pd.date_range('2020-01-01', periods=len(sample_df))
    with patch("matplotlib.pyplot.show"):
        plot_polar_stereo(sample_df, mode='aggregate', agg_func='mean')
        plt.close()

    with patch("matplotlib.pyplot.show"):
        plot_polar_stereo(sample_df, mode='subplot', time_col='time', agg_func='mean')
        plt.close()

    # should have an overlay
    with patch("matplotlib.pyplot.show"):
        plot_polar_stereo(sample_df, mode='overlay', time_col='time', max_subplots=4)
        plt.close()

    # test saving
    save_path = tmp_path / "polar.png"
    sample_df['time'] = pd.date_range('2020-01-01', periods=len(sample_df))
    with patch("matplotlib.pyplot.show"):
        plot_polar_stereo(sample_df, mode='animate', time_col='time', savefig=str(save_path))
        plt.close()
    assert save_path.exists()

    with patch("matplotlib.pyplot.show"):
        plot_polar_stereo(sample_df, mode='overlay', time_col='time', savefig=str(save_path))
        plt.close()
    assert save_path.exists()


def test_plot_polar_stereo_invalid(sample_df):
    with patch("matplotlib.pyplot.show"):
        with pytest.raises(ValueError, match='time_col required'):
            plot_polar_stereo(sample_df, mode='animate')


    with patch("matplotlib.pyplot.show"):
        with pytest.raises(ValueError, match='Missing required'):
            plot_polar_stereo(sample_df, mode='animate', time_col='not_a_col')





