from stellar_harvest_ie_config.utils.log_decorators import log_io

from stellar_harvest_ie_ml_stellar.data.loader import load_planetary_kp_index
from stellar_harvest_ie_ml_stellar.features.build_features import extract_feature_labels_planetary_kp_index
from stellar_harvest_ie_ml_stellar.models.classification.validate import validate
from stellar_harvest_ie_ml_stellar.models.classification.train import train
from stellar_harvest_ie_ml_stellar.models.classification.predict import predict
from stellar_harvest_ie_ml_stellar.models.classification.evaluate import evaluate


@log_io()
def run_classification_pipeline(url: str) -> dict:
    df = load_planetary_kp_index(url=url)
    validate()
    X, y = extract_feature_labels_planetary_kp_index(df=df)
    model, X_train, X_test, y_train, y_test = train(X=X, y=y)
    predict(model=model, X_test=X_test)
    model_metrics = evaluate(model=model, X_test=X_test, y_test=y_test)

    model_metrics
