import yaml

from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel

from stellar_harvest_ie_config.utils.log_decorators import log_io


class AppConfig(BaseModel):
    package_name: str


class ModelConfig(BaseModel):
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: Optional[int] = None

    features_raw: List[str]
    features_transformed: List[str]
    features_categorical: List[str]
    target: str


class Config(BaseModel):
    app_config: AppConfig
    model_cfg: ModelConfig


@log_io()
def load_config(config_path: Path = None) -> Config:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yml"

    with config_path.open("r") as file:
        raw_config = yaml.safe_load(file)
    return Config.model_validate(raw_config)


config = load_config()
