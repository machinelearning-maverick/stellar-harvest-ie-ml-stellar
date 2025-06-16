from stellar_harvest_ie_ml_stellar.models.classification.config.core import load_config


def test_load_config_returns_valid_object():
    config = load_config()

    assert config.app_config.package_name == "stellar-harvest-ie-ml-stellar"
    assert isinstance(config.model_config.features, list)
    assert isinstance(config.model_config.target, str)
    assert config.model_config.test_size > 0
    assert config.model_config.n_estimators > 0
