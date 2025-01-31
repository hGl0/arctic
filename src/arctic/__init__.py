# necessary libraries
# could be improved?
import numpy as np
from scipy.cluster.hierarchy import dendrogram

##### Constants #####
# 0°C in K
ZERO_DEG = 273.15

# specific gas constant for dry air (J/(kg*K), with 28.964917 g/mol
R = 287.052874
# specific gas constant used by U.S. Standard Atmosphere (J/(K*mol)
# not recommended to use due to inconsistency w.r.t. Avogadro constant and Boltzmann constant
R_ussa = 8.31432

# acceleration due to gravity (m/s^2)
g = 9.80665

# Pressure P_0 at sea level (hPa)
P0 = 1013.25


def plot_dendrogram(model, **kwargs):
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
    dendrogram(linkage_matrix, **kwargs)
