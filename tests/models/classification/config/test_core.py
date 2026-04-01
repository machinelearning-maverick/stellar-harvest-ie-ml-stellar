from stellar_harvest_ie_ml_stellar.models.classification.config.core import load_config


def test_load_config_returns_valid_object():
    config = load_config()

    assert config.app_config.package_name == "stellar-harvest-ie-ml-stellar"
    assert isinstance(config.model_cfg.features_raw, list)
    assert isinstance(config.model_cfg.target, str)
    assert config.model_cfg.test_size > 0
    assert config.model_cfg.n_estimators > 0
