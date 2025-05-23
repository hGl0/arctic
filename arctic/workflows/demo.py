import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_ssa_grid(data_series, ssa_results, labels,
                  index_format=['%b %d'],
                  figsize=(15, 8),
                  used_signals = 1,
                  titles = None):
    """
    Plot a grid of original vs SSA seasonal component time series.

    Parameters:
    - data_series: list of original time series (e.g., [avg_year, avg_day_over_year])
    - ssa_results: list of SSA seasonal components (same shape as data_series)
    - labels: list of labels (same shape: [ [label1, label2, ...], [label1, label2, ...] ])
    - index_format: datetime x-axis tick format, default is month-day
    - figsize: overall figure size
    - used_signals: number of SSA components to reconstruct signal
    - titles: list of titles for each subplot
    """
    m = len(data_series)
    n = len(labels)

    fig, ax = plt.subplots(m, n, figsize=figsize, squeeze=False)

    for i in range(m):
        if len(index_format) > 1:
            format = index_format[i]
        for j in range(n):
            ax_ij = ax[i][j]

            ts = data_series[i]
            ssa = ssa_results[i][j]
            label = labels[j]

            if used_signals > ssa[:].shape[0]:
                print(f"Number of given signals too high. Reset to maximum number {ssa.shape[0]} of signals in SSA.")
                used_signals = ssa.shape[0]
            if used_signals > 1:
                reconstructed = ssa[:][:used_signals].sum(axis=0)

            seasonal = ssa[:][0]

            if titles:
                title = str(label) + ' ' + titles[i]
            else:
                title = None

            ax_ij.plot(ts.index, ts[label], label='Averaged data')
            ax_ij.plot(ts.index, seasonal, label='Trend', linestyle='--')
            if used_signals > 1: ax_ij.plot(ts.index, reconstructed, label='Reconstructed', linestyle='-.')

            if title: ax_ij.set_title(title)
            ax_ij.xaxis.set_major_formatter(mdates.DateFormatter(format))
            ax_ij.tick_params(axis='x', rotation=45)
            ax_ij.legend()

    plt.tight_layout()
    plt.show()
