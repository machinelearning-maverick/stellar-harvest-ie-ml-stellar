from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import numpy as np
import pandas as pd

from stellar_harvest_ie_config.utils.log_decorators import log_io


@log_io(
    skip_types_input={
        pd.Series: lambda v: f"<Series name={v.name} len={len(v)}>",
        np.ndarray: lambda v: f"<ndarray shape={v.shape} dtype={v.dtype}>",
    }
)
def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    model_metrics: dict = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "class_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return model_metrics
