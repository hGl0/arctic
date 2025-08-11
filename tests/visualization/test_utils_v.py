import logging
from unittest import mock
from unittest.mock import patch
import matplotlib.pyplot as plt
import pytest
import pandas as pd
import numpy as np

from vortexclust.visualization.utils import feature_consistence, create_animation, create_polar_ax, plot_ellipse

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
    with mock.patch("matplotlib.pyplot.show"):
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