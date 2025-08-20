from pathlib import Path

from retriever.config.general import GeneralConfig
from retriever.config.openapi import OpenAPIConfig


def write_default_configs() -> None:
    """Write out config defaults."""
    GeneralConfig.write_default(Path("config/config.default.yaml").resolve())
    OpenAPIConfig.write_default(Path("config/openapi.default.yaml").resolve())
