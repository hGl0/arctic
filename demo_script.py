from sklearn.cluster import AgglomerativeClustering

import vortexclust as vc
from vortexclust.io import *

import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

import os
import logging

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


import io, contextlib  # NEW

class _LogWriter(io.TextIOBase):  # NEW
    def __init__(self, logger, level=logging.INFO):
        self.logger, self.level, self._buf = logger, level, ""
    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line.rstrip())
        return len(s)
    def flush(self):
        if self._buf.strip():
            self.logger.log(self.level, self._buf.rstrip())
        self._buf = ""

def set_up_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"vortexclust_script_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
        force=True,  # important if the script is rerun (e.g., in notebooks)
    )
    return logging.getLogger(__name__)  # return a module logger

def main():
    # CHANGE paths accordingly to your own demo data
    demo_d = "data/demo_d.csv"
    demo_msw = "data/demo_msw.csv"
    # CHANGE to save plots elsewhere
    output_dir = "output/demo/"

    # CHANGE accordingly to attributes where seasonality should be filtered
    to_filter = ["scaled_u"]
    # CHANGE accordingly to window size in Singular Spectrum Analysis
    M = 120
    # CHANGE to number of components that should be used for filtering
    n_components = 4
    # CHANGE to display plots during runtime
    show_plots = False
    # CHANGE to desired number of clusters, can be None and is determined by the gap statistic
    k_opt = 3

    set_up_logging(log_dir='logs')

    # ---------- Start of script --------
    df_d = read_data(demo_d)
    df_msw = read_data(demo_msw)

    no_white_space(df_d)
    no_white_space(df_msw)

    # unknown time format, defining explicit time format is safer
    to_date(df_d, 'string', format="mixed")
    to_date(df_msw, 'string', format="mixed")

    demo_all = df_d.merge(df_msw, on='string', how='left', suffixes=("_d", "_msw"))

    # encode stings with numeric labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    demo_all['form'] = le.fit_transform(demo_all['form'])
    logging.info(f"Transformed 'form':"
                 f"{le.inverse_transform([1])} to 1;"
                 f"{le.inverse_transform([0])} to 0")

    demo_all = demo_all.sort_values('string').reset_index(drop=True)

    # reduce to winter month
    demo = demo_all[demo_all['string'].dt.month.isin([12,1,2,3])]
    demo.reset_index(drop=True, inplace=True)

    # Scale data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    scaled_col = ['scaled_area', 'scaled_ar', 'scaled_latcent', 'scaled_u']
    col = ['area', 'ar', 'latcent', 'u']
    demo[scaled_col] = sc.fit_transform(demo[col])

    from vortexclust.workflows.demo import plot_timeseries_moments
    try:
        if show_plots:
            plot_timeseries_moments(demo, scaled_col, col,
                                title='Vortex Geoemtric Moments',
                                time_span=500, savefig=output_dir+"moments.png")
    except Exception as e:
        print("Exception:", e)

    # detect seasonality
    from statsmodels.stats.stattools import durbin_watson
    import statsmodels.api as sm
    seasonality_check = []
    for c in scaled_col:
        # fit multiple linear regression model
        model = sm.OLS(demo[c], demo[[col for col in scaled_col if col != c]]).fit()
        dw = durbin_watson(model.resid)
        if (dw < 1.5) or (dw > 2.5):
            logging.info(f"Seasonality check for '{c}' recommended. Durbin-Watson: {dw}")
            seasonality_check.append(c)

    from statsmodels.graphics.tsaplots import plot_acf
    for c in seasonality_check:
        plot_acf(demo[c], lags=500, marker=None)
        plt.axhline(y=0.05, linestyle='--', color='black')
        plt.axhline(y=-0.05, linestyle='--', color='black')
        plt.title(f"Autocorrelation check for '{c}'")
        plt.savefig(output_dir + f"seasonality_check_{c}.png")
        if show_plots: plt.show()

    # filter seasonality
    from pyts.decomposition import SingularSpectrumAnalysis
    ssa = SingularSpectrumAnalysis(window_size=M)
    for col in to_filter:
        ssa_computed = ssa.fit_transform(demo[col].values.reshape(1, -1))
        epc, eeof, expl_var_ratio, eeof_computed, _ = vc.compute_eeof(demo[col], M=400, n_components=n_components)
        if len(to_filter) < 2:
            from vortexclust.workflows.demo import plot_eeof
            plot_eeof(epc, eeof, expl_var_ratio)

        plt.figure()
        plt.plot(demo[col][:500], label='original')
        plt.plot(ssa_computed[:n_components].sum(axis=0)[:500], label=f"SSA with {n_components} components")
        plt.plot(eeof_computed[399:899, 0], label=f'EEOF with {n_components} components')
        plt.title(f"Singular Spectrum Analysis and EEOF for '{col}'")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"eeof_ssa_{col}.png"))
        if show_plots: plt.show()


        demo["ssa_"+col] = demo[col] - ssa_computed[:n_components].sum(axis=0)

        demo.loc[:, "eeof_"+col] = np.full_like(demo["ssa_"+col], np.nan)
        demo.loc[:309, "eeof_"+col] = demo.loc[:309, col] - eeof_computed[399:709, 0].T
        demo.loc[310:1000, "eeof_"+col] = demo.loc[310:1000, col] - eeof_computed[399:1090, 309].T

        demo = demo[:1000]

        plt.figure()
        plt.plot(demo.loc[:500, col], label='original')
        plt.plot(demo.loc[:500, "ssa_"+col], label="SSA filtered")
        plt.plot(demo.loc[:500, "eeof_"+col], label='EEOF filtered')
        plt.title("Filtered signal")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"filtered_{col}.png"))
        if show_plots: plt.show()

    # determine k_opt
    if k_opt is None:
        k_max = 10
        gap_ar_latcent_u = vc.gap_statistic(demo[['scaled_ar', 'scaled_latcent', 'ssa_scaled_u']], k_max = k_max, n_replicates=10)

        plt.errorbar(np.arange(1, k_max+1), gap_ar_latcent_u[:, 0], y_err=gap_ar_latcent_u[:, 1], label="AR, Latcent, Wind Speed")
        plt.savefig(output_dir+"/gap_ar_latcent_u.png")
        if show_plots: plt.show()

        p = 1
        for k in range(1, 10):
            if p and (gap_ar_latcent_u[k][0] >= gap_ar_latcent_u[k + 1][0] - gap_ar_latcent_u[k + 1][1]):
                logging.info("Gap statistic (AR, Latcent, filtered wind speed): ", k + 1)
                k_opt = k+1
                break

    model = AgglomerativeClustering(linkage='complete', compute_distances=True, n_clusters=k_opt)
    model.fit(demo[['scaled_ar', 'scaled_latcent', 'eeof_scaled_u']])
    y = model.labels_.astype(int)
    demo['y'] = y


    # visualisation model
    import vortexclust.visualization as viz
    viz.plot_dendrogram(model, truncate_mode='level', p=4, direction='LR', savefig=output_dir+"demo_dendrogram.png")

    # statistics of classes
    from vortexclust.workflows.demo import plot_hist_per_class
    logging.info(f"Averages per class:\n"
                 f"{demo[['y', 'scaled_ar', 'scaled_latcent', 'eeof_scaled_u']].groupby(['y']).mean()}")
    plot_hist_per_class(demo, {'features' : ['scaled_ar', 'scaled_latcent', 'eeof_scaled_u']}, 'y', savefig=output_dir+"demo_hist_per_class")

    # compare to form
    from vortexclust.workflows.demo import compare_cluster

    logger = logging.getLogger(__name__)  # use the logger configured above

    # Redirect stdout/stderr -> logging only for this section
    with contextlib.redirect_stdout(_LogWriter(logger, logging.INFO)), \
            contextlib.redirect_stderr(_LogWriter(logger, logging.ERROR)):
        compare_cluster(
            demo,
            compare_col='form',
            pred_value=1,
            gt_value=1,
            y_names=['y']
        )


if __name__ == "__main__":
    main()
