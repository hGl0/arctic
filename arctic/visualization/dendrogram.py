from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

from arctic.io.paths import check_path


def plot_dendrogram(model: Any,
                    **kwargs) -> None:
    r"""
    Plots a dendrogram for a given hierarchical clustering model with computed distances between samples.

    :param model: A clustering model that must have 'children_', 'distances_', and 'labels_' attributes.
    :param kwargs: Additional arguments for 'scipy.cluster.hierarchy.dendrogram'.
                - `savefig` (str, optional): Path to save the figure. Raises an error if the directory does not exist.

    :raises AttributeError: If 'model' is missing required attributes.
    :raises TypeError: If 'model' attributes are of incorrect data types.
    :raises FileNotFoundError: If the directory for `savefig` does not exist.
    :raises Exception: For unexpected errors.

    :return: None
    """

    save_path = kwargs.pop('savefig', None)
    direction = kwargs.pop('direction', None)


    # check availability of attributes
    if not all(hasattr(model, attr) for attr in ['children_', 'distances_', 'labels_']):
        raise AttributeError("Model must have 'children_', 'distances_', and 'labels_' attributes.")

    orientation_map = {
        "TB": "top",
        "BT": "bottom",
        "LR": "left",
        "RL": "right"
    }
    orientation = orientation_map.get(direction.upper())
    # if orientation is None:
    #     raise ValueError(f"Invalid direction '{direction}'. Choose from: {list(orientation_map.keys())}")

    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    if direction is not None:
        dendrogram(linkage_matrix, orientation=orientation, **kwargs)
    else:
        dendrogram(linkage_matrix, **kwargs)

    # save the dendrogram
    if save_path:
        # Validate savefig
        check_path(save_path)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
