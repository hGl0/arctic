from pathlib import Path

# === Project Root === #
ROOT_DIR = Path(__file__).resolve().parent.parent

# === Directory Paths === #
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
DOC_DIR = ROOT_DIR / "doc"

# Optional: subfolders
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = OUTPUT_DIR / "figures"
ANIMATIONS_DIR = OUTPUT_DIR / "animations"

# Ensure folders exist (optional)
for d in [OUTPUT_DIR, FIGURES_DIR, ANIMATIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Global Constants === #
DEFAULT_TIMEZONE = "UTC"
DEFAULT_LATITUDE_BAND = (50, 90)  # Degrees North, e.g. Arctic region
DEFAULT_PRESSURE_LEVELS = [10, 50, 100]  # hPa for stratosphere

# === Plotting Defaults === #
PLOT_STYLE = "seaborn-whitegrid"
COLOR_PALETTE = "viridis"
DPI = 150
FIGSIZE = (10, 10)

# === Logging Setup === #
import logging

LOG_LEVEL = logging.INFO
LOG_FORMAT = "[%(levelname)s] %(asctime)s - %(message)s"
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

logger = logging.getLogger("vortexclust")

# === Environment Flags === #
DEBUG_MODE = False
VERBOSE = True

# === Utility Functions === #
def set_debug_mode(enabled: bool = True):
    global DEBUG_MODE
    DEBUG_MODE = enabled
    logger.setLevel(logging.DEBUG if enabled else LOG_LEVEL)
    logger.debug("Debug mode enabled." if enabled else "Debug mode disabled.")

