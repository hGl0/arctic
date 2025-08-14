# vortexclust/workflows/__init__.py
import importlib
from importlib.util import find_spec

__all__ = ["demo"]

def __getattr__(name):
    if name == "demo":
        try:
            # Try importing the submodule only when requested
            return importlib.import_module(".demo", __name__)
        except ImportError as e:
            # Check if seaborn (or any other demo-only dep) is available
            missing_demo_dep = find_spec("seaborn") is None
            if missing_demo_dep:
                raise RuntimeError(
                    "The demo workflow requires optional dependencies. "
                    "Install them via: pip install vortexclust[demo]"
                ) from e
            # If seaborn exists but import still failed, re-raise the original error
            raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
