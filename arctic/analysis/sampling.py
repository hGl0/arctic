import numpy as np

def generate_reference_HPPP(X: np.ndarray) -> np.ndarray:
    r"""
    Generates a reference dataset using a Homogeneous Poisson Point Process (HPPP).

    :param X: Input data.
    :type X: numpy.ndarray

    :raises ValueError: If `X` contains non-numeric values.

    :return: ndarray containing a randomly generated reference data matching the shape and range of 'X'
    :rtype: numpy.ndarray
    """
    # check if X is numeric
    return np.random.uniform(low=X.min(axis=0), high=X.max(axis=0), size=X.shape)