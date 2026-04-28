import numpy as np

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.models.regression import __version__ as _version


@log_io(skip_types={
    np.ndarray: lambda v: f"<ndarray shape={v.shape} dtype={v.dtype}>",
})
def predict(model, X_test) -> dict:
    y_pred = model.predict(X_test)
    results = {"predictions": y_pred, "version": _version, "validation_errors": None}

    return results
