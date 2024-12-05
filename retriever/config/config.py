from pathlib import Path
from pydantic import BaseModel, BaseSettings
from pydantic.env_settings import SettingsSourceCallable

from retriever.config.util import get_yaml_settings


class RateLimit(BaseModel):
    special: int = 100 * 60
    general: int = 600


class GeneralConfig(BaseSettings):
    rate_limit: RateLimit = RateLimit()

    class Config:
        case_sensitive = False
        env_nested_delimiter = "__"
        env_file = ".env"
        env_file_encoding = "utf-8"

        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ):
            return (
                env_settings,
                get_yaml_settings(Path(__file__).parent / "config.yaml"),
                init_settings,
                file_secret_settings,
            )
