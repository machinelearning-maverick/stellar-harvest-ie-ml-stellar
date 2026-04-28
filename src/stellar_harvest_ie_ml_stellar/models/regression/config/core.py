import yaml

from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel, field_validator

from stellar_harvest_ie_config.utils.log_decorators import log_io


class AppConfig(BaseModel):
    package_name: str


class ModelConfig(BaseModel):
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: Optional[int] = None
    resample_rule: str
    horizon: int
    lags: List[int]
    input_features: List[str]
    target: str

    @field_validator("test_size")
    @classmethod
    def test_size_in_range(cls, v: float) -> float:
        if not (0 < v < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {v}")
        return v

    @field_validator("horizon")
    @classmethod
    def horizon_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"horizon must be >= 1, got {v}")
        return v

    @field_validator("lags")
    @classmethod
    def lags_positive_unique(cls, v: List[int]) -> List[int]:
        if not v or any(x < 1 for x in v) or len(set(v)) != len(v):
            raise ValueError(f"lags must be positive and unique, got {v}")
        return v


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
