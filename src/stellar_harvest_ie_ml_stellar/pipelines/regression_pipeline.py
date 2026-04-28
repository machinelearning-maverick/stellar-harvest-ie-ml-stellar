from stellar_harvest_ie_config.utils.log_decorators import log_io


from stellar_harvest_ie_ml_stellar.data.loader import load_planetary_kp_index
from stellar_harvest_ie_ml_stellar.models.regression.validate import validate
from stellar_harvest_ie_ml_stellar.models.regression.features import extract
from stellar_harvest_ie_ml_stellar.models.regression.train import train
from stellar_harvest_ie_ml_stellar.models.regression.predict import predict
from stellar_harvest_ie_ml_stellar.models.regression.evaluate import evaluate


@log_io()
async def run_regression_pipeline() -> dict:
    df = await load_planetary_kp_index()
    validate(df=df)
    X, y = extract(df=df)
    model, X_train, X_test, y_train, y_test = train(X=X, y=y)
    predict(model=model, X_test=X_test)
    model_metrics = evaluate(model=model, X_test=X_test, y_test=y_test)

    return model_metrics
