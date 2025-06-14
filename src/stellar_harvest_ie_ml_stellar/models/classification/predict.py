import pandas as pd

from stellar_harvest_ie_config.utils.log_decorators import log_io


@log_io()
def predict(model, X_test) -> dict:
    y_pred = model.predict(X_test)
    results = {"predictions": y_pred, "version": 1, "validation_errors": None}

    return results
