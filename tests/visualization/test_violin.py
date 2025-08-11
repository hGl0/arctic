import logging
from unittest.mock import patch
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

from vortexclust.visualization.violin import plot_violin

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

def test_plot_violin(sample_df, tmp_path):
    # Basic test without saving
    with patch('matplotlib.pyplot.show'):
        plot_violin(sample_df)
        plt.close()

    # Save figure
    fig_path = tmp_path / "violin.png"
    with patch('matplotlib.pyplot.show'):
        plot_violin(sample_df, savefig=str(fig_path))
        plt.close()
    assert fig_path.exists()

    # Custom features
    features = ['feature1', 'feature2', 'feature3']
    with patch('matplotlib.pyplot.show'):
        plot_violin(sample_df, features=features, n_features=3)
        plt.close()

    # Custom scaler
    with patch('matplotlib.pyplot.show'):
        plot_violin(sample_df, scaler=RobustScaler)
        plt.close()

def test_plot_violin_invalid(sample_df):
    df = sample_df.rename(columns={'label': 'invalid_label'})
    with pytest.raises(KeyError):
        plot_violin(df)


    df = sample_df.copy()
    df['featureX'] = ['non-numeric'] * 10
    with pytest.raises(TypeError, match="Ensure your dataframe has only numeric types."):
        plot_violin(df)




