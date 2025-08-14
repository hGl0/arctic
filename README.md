# Clustering of stratospheric polar vortex regimes
`Vortexclust` is a python package to analyse and cluster climate data. It has a special focus on the segmentation and influences in the stratospheric polar vortex and sudden stratospheric warmings.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [FAQ](#faq)
- [Acknowledgments](#acknowledgments)

## Installation <a name="installation"></a>
### Install with pip (Python >= 3.9 required)

Core package:<br>
`pip install vortexclust`

With optional dependencies: <br>
Adds cartopy/pyproj (needs system GEOS/PROJ) and is only required if `vortexclust.visualization.maps` is used to generate a stereographic map plot<br>
`pip install vortexclust[maps]`

#### From GitHub
`pip install git+https://github.com/hGl0/vortexclust@main`

#### From local source
```
git clone https://github.com/hGl0/vortexclust.git
cd vortexclust
pip install -e .
```

### Windows Installation
It is recommended to install `miniconda` or `anaconda`. This handles all Python dependencies without manual compilation.
Example:
```
conda create -n vortex python=3.11
conda activate vortex
pip install vortexclust[viz, maps]
```

## Getting started
Load necessary libraries
```
import vortexclust as vc
from vortexclust.io import *

import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings("ignore")

```

Set up variables to read data, cleaning white space and convert to correct types.
```
# CHANGE paths accordingly to your own demo data
demo_d = "data/demo_d.csv"
demo_msw = "data/demo_msw.csv"

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
print(f"Transformed 'form':"
             f"{le.inverse_transform([1])} to 1;"
             f"{le.inverse_transform([0])} to 0")

demo_all = demo_all.sort_values('string').reset_index(drop=True)

# reduce to winter month
demo = demo_all[demo_all['string'].dt.month.isin([12,1,2,3])]
demo.reset_index(drop=True, inplace=True)
```

Scale the data using StandardScaler. The data need to be reduced to a respective time range *before* scaling.
```
output_dir = "output/demo/"

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_col = ['scaled_area', 'scaled_ar', 'scaled_latcent', 'scaled_u']
col_to_scale = ['area', 'ar', 'latcent', 'u']
demo[scaled_col] = sc.fit_transform(demo[col_to_scale])

from vortexclust.workflows.demo import plot_timeseries_moments
try:
    plot_timeseries_moments(demo, scaled_col, col_to_scale,
                            title='Vortex Geoemtric Moments',
                            time_span=500, savefig=output_dir+"moments.png")
except Exception as e:
    print("Exception:", e)
```

Some data contain seasonality which should be filtered before clustering. Below, the autocorrelation function is plotted.
Additionally, the Durbin-Watson test can be used to detect autocorrelation.
```
from statsmodels.graphics.tsaplots import plot_acf
for c in scaled_col:
    plot_acf(demo[c], lags=500, marker=None)
    plt.axhline(y=0.05, linestyle='--', color='black')
    plt.axhline(y=-0.05, linestyle='--', color='black')
    plt.title(f"Autocorrelation check for '{c}'")
    plt.savefig(output_dir + f"seasonality_check_{c}.png")
    plt.show()
```

If seasonality is detected, it should be filtered. Seasonality often leads to inferior models 
that detect different phases instead of different regimes.
```
to_filter = ["scaled_u"]
n_components = 4

from vortexclust.workflows.demo import plot_eeof

for col in to_filter:
    # compute EEOF
    epc, eeof, expl_var_ratio, eeof_computed, _ = vc.compute_eeof(demo[col], M=400, n_components=n_components)
       
    # plot EPCs and EEOFs
    plot_eeof(epc, eeof, expl_var_ratio, savefig=output_dir+f"eeof")
    
    # plot EEOF reconstructed signals
    plt.figure()
    plt.plot(demo[col][:500], label='original')
    plt.plot(eeof_computed[399:899, 0], label=f'EEOF with {n_components} components')
    plt.title(f"EEOF for '{col}'")
    plt.legend()
    plt.savefig(output_dir+f"eeof_{col}.png")
    plt.show()
    
    demo.loc[:, "eeof_"+col] = np.full_like(demo[col], np.nan)
    demo.loc[:309, "eeof_"+col] = demo.loc[:309, col] - eeof_computed[399:709, 0].T
    demo.loc[310:1000, "eeof_"+col] = demo.loc[310:1000, col] - eeof_computed[399:1090, 309].T
    demo = demo[:1000]
    
    # plot filtered signals
    plt.figure()
    plt.plot(demo.loc[:500, col], label='original')
    plt.plot(demo.loc[:500, "eeof_"+col], label='EEOF filtered')
    plt.title("Filtered signal")
    plt.legend()
    plt.savefig(output_dir+f"filtered_{col}.png")
    plt.show()
```

The number of clusters is usually determined by external methods such as the gap statistic, silhouette or elbow method.
Here, it is set to 3 for reasons of simplicity and from prior analysis. Below a hierarchical clustering model is fitted 
to the data. Special notice should be the linkage, which computes the distance between two clusters. Some linkages tend 
to produce chained clusters and reinforce the addition to large clusters. To avoid such behaviour *complete* or *ward* 
is recommended to use.
```
k_opt = 3
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(linkage='complete', compute_distances=True, n_clusters=k_opt)
model.fit(demo[['scaled_ar', 'scaled_latcent', 'eeof_scaled_u']])
y = model.labels_.astype(int)
demo['y'] = y
# visualisation model
import vortexclust.visualization as viz
viz.plot_dendrogram(model, truncate_mode='level', p=4, direction='LR', savefig=output_dir+"demo_dendrogram.png")
```

Finally, the descriptive statistics of each cluster help to understand it behaviour, formation and characteristics.
```
# statistics of classes
from vortexclust.workflows.demo import plot_hist_per_class
print(f"Averages per class:\n"
             f"{demo[['y', 'scaled_ar', 'scaled_latcent', 'eeof_scaled_u']].groupby(['y']).mean()}")

plot_hist_per_class(demo, {'features' : ['scaled_ar', 'scaled_latcent', 'eeof_scaled_u']}, 'y', savefig=output_dir+"demo_hist_per_class")

# compare to form
from vortexclust.workflows.demo import compare_cluster
compare_cluster(
    demo,
    compare_col='form',
    pred_value=1,
    gt_value=1,
    y_names=['y']
)
```

## Usage <a name="usage"></a>
[To Do] Short discription of usage, reference to jupyternotebook as user story

## Tests
[To Do] Description how to start tests

## Contributing <a name="contributing"></a>
[To Do] How to contribute, whom to contact

## License <a name="license"></a>
This project is licensed under the [GPL3.0](LICENSE).

## FAQ <a name="faq"></a>

## Acknowledgments <a name="acknowledgments"></a>
