import yaml

from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

from stellar_harvest_ie_config.utils.log_decorators import log_io


class AppConfig(BaseModel):
    package_name: str


class ModelConfig(BaseModel):
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: Optional[int] = None

    input_features: List[str]
    features_raw: List[str]
    features_transformed: List[str]
    features_categorical: List[str]
    target: str

    @field_validator("test_size")
    @classmethod
    def test_size_in_range(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {v}")
        return v

    @field_validator("n_estimators")
    @classmethod
    def n_estimators_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"n_estimators must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def features_consistency(self) -> "ModelConfig":
        raw_set = set(self.features_raw)
        invalid = set(self.features_categorical) - raw_set
        if invalid:
            raise ValueError(f"features_categorical not in features_raw: {invalid}")
        if self.target not in raw_set:
            raise ValueError(f"target '{self.target}' not in features_raw")
        return self


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
