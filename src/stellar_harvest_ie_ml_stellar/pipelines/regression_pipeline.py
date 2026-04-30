from stellar_harvest_ie_config.utils.log_decorators import log_io

from stellar_harvest_ie_store.db import AsyncSessionLocal
from stellar_harvest_ie_store.repository import AsyncRepository
from stellar_harvest_ie_models.stellar.swpc.entities import KpForecastEntity

from stellar_harvest_ie_ml_stellar.data.loader import load_planetary_kp_index
from stellar_harvest_ie_ml_stellar.models.regression.validate import validate
from stellar_harvest_ie_ml_stellar.models.regression.features import extract
from stellar_harvest_ie_ml_stellar.models.regression.train import train
from stellar_harvest_ie_ml_stellar.models.regression.predict import predict
from stellar_harvest_ie_ml_stellar.models.regression.evaluate import evaluate
from stellar_harvest_ie_ml_stellar.models.regression.forecast import forecast


@log_io()
async def run_regression_pipeline(n_forecast_steps: int = 8) -> dict:
    df = await load_planetary_kp_index()
    validate(df=df)
    X, y = extract(df=df)
    model, _, X_test, _, y_test = train(X=X, y=y)
    predict_result = predict(model=model, X_test=X_test)
    model_metrics = evaluate(
        X_test=X_test, y_test=y_test, y_preds=predict_result["y_preds"]
    )

    forecast_df, forecast_entities = await forecast(
        model=model, n_steps=n_forecast_steps
    )

    async with AsyncSessionLocal() as session:
        repository = AsyncRepository(KpForecastEntity, session)
        await repository.add_all(forecast_entities)
        await session.commit()

    return {
        "metrics": model_metrics,
        "forecast": forecast_df.to_dict(orient="records"),
    }
