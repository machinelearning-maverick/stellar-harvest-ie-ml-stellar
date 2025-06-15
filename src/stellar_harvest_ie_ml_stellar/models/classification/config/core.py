import yaml

from typing import List
from pathlib import Path

from pydantic import BaseModel


class AppConfig(BaseModel):
    package_name: str


class ModelConfig(BaseModel):
    target: str
    features: List[str]


class Config(BaseModel):
    app_config: AppConfig
    model_config: ModelConfig


def load_config(config_path: Path = Path("config.yml")) -> Config:
    with config_path.open("r") as file:
        raw_config = yaml.safe_load(file)
    return Config.model_validate(raw_config)
