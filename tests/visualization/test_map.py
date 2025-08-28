import logging
import pytest
import pandas as pd
from unittest.mock import patch
import matplotlib.pyplot as plt
import numpy as np

from vortexclust.visualization.map import plot_polar_stereo, create_animation, create_polar_ax, plot_ellipse

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
    save_path = tmp_path / "polar.gif"
    sample_df['time'] = pd.date_range('2020-01-01', periods=len(sample_df))
    with patch("matplotlib.pyplot.show"):
        plot_polar_stereo(sample_df, mode='animate', time_col='time', savefig=str(save_path))
        plt.close()
    assert save_path.exists()

    save_path = tmp_path/"polar.png"
    with patch("matplotlib.pyplot.show"):
        plot_polar_stereo(sample_df, mode='overlay', time_col='time', savefig=str(save_path))
        plt.close()
    assert save_path.exists()


def test_plot_polar_stereo_invalid(sample_df):
    with patch("matplotlib.pyplot.show"):
        with pytest.raises(KeyError, match='time_col required'):
            plot_polar_stereo(sample_df, mode='animate')


    with patch("matplotlib.pyplot.show"):
        with pytest.raises(KeyError, match='Missing required'):
            plot_polar_stereo(sample_df, mode='animate', time_col='not_a_col')

def test_create_animation(tmp_path):
    df = pd.DataFrame({
        'time': pd.date_range("2023-01-01", periods=2),
        'area': [1e6, 1e6],
        'ar': [1.5, 1.6],
        'theta': [30, 60],
        'loncent': [0, 10],
        'latcent': [60, 65],
        'form': [0, 1]
    })

    savegif = tmp_path / "test.gif"
    with patch("matplotlib.pyplot.show"):
        create_animation(df, time_col='time', filled=True, savegif=str(savegif), split=1)
    assert savegif.exists()

def test_create_polar_ax():
    fig, ax = create_polar_ax()
    assert fig is not None
    assert hasattr(ax, "set_extent")
    plt.close(fig)


def test_plot_ellipse():
    fig, ax = create_polar_ax()
    x = np.cos(np.linspace(0, 2 * np.pi, 100))
    y = np.sin(np.linspace(0, 2 * np.pi, 100))
    plot_ellipse(ax, x, y, 0, 60, filled=True)
    # Check if elements were added
    assert len(ax.patches) > 0 or len(ax.lines) > 0
    plt.close(fig)

def test_plot_ellipse_invalid():
    fig, ax = create_polar_ax()
    with pytest.raises(ValueError):
        plot_ellipse(ax, [], [], 0, 60)
    plt.close(fig)