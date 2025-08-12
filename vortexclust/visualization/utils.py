from typing import List, Tuple, Optional

import pandas as pd
from vortexclust.analysis.decomposition import compute_pca


def feature_consistence(df: pd.DataFrame,
                        n_features: Optional[int] = None,
                        features: Optional[List[str]] = [],
                        ) -> Tuple[int, List[str]]:
    r"""
    Determines the most influential PCA features or uses a given feature list.

    :param df: A pandas DataFrame containing the dataset.
    :param n_features: The number of influential features to select.
    :param features: A list of predefined features (optional).

    :raises ValueError: If neither `n_features` nor `features` is provided.
    :raises Exception: If the loop exceeds the number of available features in `df`.

    :return: A tuple containing:
        - The number of selected features.
        - A list with the feature names
    """
    # at least one must be given
    if (features == []) and (n_features is None):
        raise ValueError("Either 'n_features' or 'features' must be provided.")

    # n_features has valid values
    if (not n_features is None and
            ((n_features == 0) or (not isinstance(n_features, int)))):
        raise ValueError("If 'n_features' is provided, it must be an integer greater than 0.")

    if features:
        return len(features), features

    if n_features == df.shape[1]:
        return n_features, df.columns

    add_features = 0
    max_attempts = df.shape[1]
    # based on only n_features
    while True:
        if add_features > max_attempts:
            raise Exception(f"Could not select exactly {n_features} unique features from PCA:\n"
                            f"Tried increasing but reached dataset limit.")

        pca_features = (pd.DataFrame(compute_pca(df, plot_type=None,
                                    n_comp=n_features + add_features)[1])
                        .drop(['Expl_var', 'Expl_var_ratio'])
                        .idxmax()
                        .reset_index())

        unique_features = pca_features[0].unique()

        if len(unique_features) == n_features:
            return n_features, list(unique_features)
        add_features += 1
    # n_features > features



