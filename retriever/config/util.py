from pathlib import Path
from typing import Any
from pydantic import BaseSettings
import yaml


def get_yaml_settings(fname: Path):

    def yaml_config_settings_source(settings: BaseSettings) -> dict[str, Any]:
        with open(fname, "r") as file:
            return yaml.safe_load(file)

    return yaml_config_settings_source
