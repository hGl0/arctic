import logging
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from vortexclust.visualization.radar import plot_radar

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_df():
    np.random.seed(0)
    data = {
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'feature3': np.random.rand(10),
        'feature4': np.random.rand(10),
        'feature5': np.random.rand(10),
        'feature6': np.random.rand(10),
        'feature7': np.random.rand(10),
        'label': ['A'] * 5 + ['B'] * 5
    }
    return pd.DataFrame(data)

def test_plot_radar(sample_df, tmp_path):
    # default
    with patch('matplotlib.pyplot.show'):
        plot_radar(sample_df)
        plt.close()

    # with save
    fig_path = tmp_path / 'radar.png'
    with patch('matplotlib.pyplot.show'):
        plot_radar(sample_df, savefig=str(fig_path))
        plt.close()
    assert fig_path.exists()

    # custom features
    features = ['feature1', 'feature2', 'feature3']
    with patch('matplotlib.pyplot.show'):
        plot_radar(sample_df, features=features, n_features=3)
        plt.close()

def test_plot_radar_invalid(sample_df):
    df = sample_df.rename(columns={'label':'invalid_label'})
    with patch('matplotlib.pyplot.show'):
        with pytest.raises(KeyError, match="Please give a valid label"):
            plot_radar(df)


    df = sample_df.copy()
    df['f1'] = ['X']*10
    with pytest.raises(TypeError, match="Ensure your dataframe"):
        with patch('matplotlib.pyplot.show'):
            plot_radar(df)
