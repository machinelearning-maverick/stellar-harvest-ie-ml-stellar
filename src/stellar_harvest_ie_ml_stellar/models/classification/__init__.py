from pathlib import Path
from stellar_harvest_ie_ml_stellar.utils.model_version import read_version

__version__ = read_version(Path(__file__).resolve().parent)
