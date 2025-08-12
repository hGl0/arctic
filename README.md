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

With optional plotting extras:<br>
Adds seaborn: `pip install vortexclust[viz]`<br>
Adds cartopy/pyproj (needs system GEOS/PROJ): `pip install vortexclust[maps]`

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

[To Do] snippets to get started: import, load demo csv, cluster plot or function


## Usage <a name="usage"></a>
[To Do] Short discription of usage, reference to jupyternotebook as user story

## Contributing <a name="contributing"></a>
[To Do] How to contribute, whom to contact

## License <a name="license"></a>
This project is licensed under the [GPL3.0](LICENSE).

## FAQ <a name="faq"></a>

## Acknowledgments <a name="acknowledgments"></a>
