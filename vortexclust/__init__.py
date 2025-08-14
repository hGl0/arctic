# Key functionalities
from .core.constants import *
from .analysis import *

__version__ = "0.1.2" # demo workflows included
__author__ = "Hanna Gloyna"

import importlib

def __getattr__(name):
    if name == "visualization":
        # Importing the subpackage is cheap now; heavy deps are handled inside.
        return importlib.import_module(f".visualization", __name__)
    if name == "workflows":
        return importlib.import_module(f".workflows", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
