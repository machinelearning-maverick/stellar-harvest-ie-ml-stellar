import pandas as pd

from stellar_harvest_ie_config.utils.log_decorators import log_io


@log_io()
def load_planetary_kp_index(url: str) -> pd.DataFrame:
    return pd.read_json(url)
