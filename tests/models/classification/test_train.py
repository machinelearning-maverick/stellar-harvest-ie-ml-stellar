import pandas as pd

from stellar_harvest_ie_ml_stellar.models.regression.train import train


def test_train_function_runs():
    df = pd.DataFrame(
        {
            "kp": ["1M", "2Z", "1M", "0P"],
            "estimated_kp": [2.0, 3.1, 1.5, 0.5],
            "year": [2025] * 4,
            "month": [6] * 4,
            "day": [15] * 4,
            "hour": [12, 13, 14, 15],
            "minute": [0, 0, 0, 0],
        }
    )

    y = pd.Series([0, 2, 0, 1])

    model, X_train, X_test, y_train, y_test = train(df, y)

    from sklearn.ensemble import RandomForestClassifier

    assert isinstance(model, RandomForestClassifier)

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
