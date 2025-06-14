import pandas as pd

from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from stellar_harvest_ie_config.utils.log_decorators import log_io


@log_io()
def train(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )

    kp_known = sorted(X_train["kp"].unique())
    kp_unknown = "<UNKNOWN>"
    kp_all = [kp_known + [kp_unknown]]

    X_train["kp"] = X_train["kp"].where(X_train["kp"].isin(kp_known), kp_unknown)
    X_test["kp"] = X_test["kp"].where(X_test["kp"].isin(kp_known), kp_unknown)

    # Categorical data
    categorical_features = ["kp"]

    one_hot_enc = OneHotEncoder(
        categories=kp_all, drop=None, sparse_output=False, handle_unknown="ignore"
    )  # unseen -> all-zero except the <UNKNOWN> column

    transformer = ColumnTransformer(
        [("one_hot_enc", one_hot_enc, categorical_features)],
        remainder="passthrough",
        verbose_feature_names_out=False,
        sparse_threshold=0.0,
    )

    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train_transformed, y_train)

    return model, X_train_transformed, X_test_transformed, y_train, y_test
