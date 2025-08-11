import numpy as np
import pandas as pd

def generate_reference_HPPP(X: np.ndarray) -> np.ndarray:
    r"""
    Generates a reference dataset using a Homogeneous Poisson Point Process (HPPP).

    :param X: Input data.
    :type X: numpy.ndarray

    :raises ValueError: If `X` contains non-numeric values.

    :return: ndarray containing a randomly generated reference data matching the shape and range of 'X'
    :rtype: numpy.ndarray
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        data = X.values
    else:
        data = X

    if not np.issubdtype(data.dtype, np.number):
        raise ValueError("Input data must be numeric.")

    return np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=X.shape)