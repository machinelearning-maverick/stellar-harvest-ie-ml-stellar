import numpy as np

from stellar_harvest_ie_config.utils.log_decorators import log_io
from stellar_harvest_ie_ml_stellar.models.regression import __version__ as _version


@log_io(
    skip_types_input={
        np.ndarray: lambda v: f"<ndarray shape={v.shape} dtype={v.dtype}>",
    }
)
def predict(model, X_test) -> dict:
    y_pred = np.clip(model.predict(X_test), 0.0, 9.0)
    return {"predictions": y_pred, "version": _version, "validation_errors": None}
