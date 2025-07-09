import logging

import pandas as pd
import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from arctic.analysis import *

logger = logging.getLogger(__name__)

@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame(np.random.rand(100, 5),
                        columns=[f'feature_{i}' for i in range(5)])

@pytest.mark.parametrize("scaler", [StandardScaler(),
                                    MinMaxScaler(),
                                    RobustScaler()])
def test_compute_pca(sample_df, tmp_path, scaler):
    # generell
    X_pca, scores, pca_obj = compute_pca(sample_df, n_comp=3)
    assert isinstance(X_pca, np.ndarray)
    assert isinstance(scores, pd.DataFrame)
    assert X_pca.shape[0] == sample_df.shape[0]
    assert X_pca.shape[1] >= 3
    assert scores.shape[1] == 3
    assert hasattr(pca_obj, 'components_')

    # with scaling
    X_pca, _, _ = compute_pca(sample_df, n_comp=3, scaler=scaler) # valid scaler class
    assert isinstance(X_pca, np.ndarray)
    X_pca, _, _ = compute_pca(sample_df, n_comp=3, scaler="invalid") # not a scaler
    assert isinstance(X_pca, np.ndarray)
    sc = StandardScaler()
    X_pca, _, _ = compute_pca(sample_df, scaler=sc) # invalid scaler object
    assert isinstance(X_pca, np.ndarray)

    # with saving
    path = tmp_path / 'pca.csv'
    compute_pca(sample_df, n_comp=3, savecsv=str(path))
    assert path.exists()
    saved = pd.read_csv(path)
    assert not saved.empty
    assert saved.shape[1]  == 3+1 # explained variance added

    #

def test_compute_pca_invalid(sample_df):
    # components exceeds feature
    with pytest.warns(UserWarning, match="Less components than given"):
        _, scores, _ = compute_pca(sample_df, n_comp=10)
    assert scores.shape[1] == sample_df.shape[1] # should cap at 5
    # invalid dataframe
    with pytest.raises(TypeError, match="Input data must"):
        compute_pca("not a DataFrame")





def test_compute_eeof():
    signal = 1.5*np.sin(np.linspace(0, 4*np.pi, 500)) + np.random.rand(500)
    epcs, eeofs, eigenvalues, reconstructed, delay = compute_eeof(signal, M=100, n_components=5)
    assert isinstance(epcs, np.ndarray)
    assert isinstance(eeofs, np.ndarray)
    assert isinstance(eigenvalues, np.ndarray)
    assert isinstance(reconstructed, np.ndarray)
    assert isinstance(delay, np.ndarray)
    assert epcs.shape[1] == 5
    assert eeofs.shape[0] == 5
    assert eigenvalues.shape[0] == 5
    assert reconstructed.shape[1] == 100


def test_compute_eeof_invalid_window():
    signal = np.random.rand(100)
    with pytest.raises(ValueError, match="Time series length must be greater"):
        compute_eeof(signal, M=200)