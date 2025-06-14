import pandas as pd

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.models.classification import __version__ as _version


@log_io()
def predict(model, X_test) -> dict:
    y_pred = model.predict(X_test)
    results = {"predictions": y_pred, "version": _version, "validation_errors": None}

    return results
