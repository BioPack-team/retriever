from typing import Annotated, ClassVar, override

from pydantic import AfterValidator, BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from retriever.types.general import LogLevel


class CORSSettings(BaseModel):
    """CORS-specific settings."""

    allow_origins: list[str] = ["*"]
    allow_credentials: bool = True
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]


class RedisSettings(BaseModel):
    """Redis client initialization settings."""

    cluster: bool = False
    host: str = "localhost"
    port: int = 6379
    password: str | None = None
    ssl_enabled: bool = False
    attempts: int = 3


class MongoSettings(BaseModel):
    """Mongodb client initialization settings."""

    host: str = "localhost"
    port: int = 27017
    username: str | None = None
    password: str | None = None
    authsource: str | None = None
    attempts: int = 3
    shutdown_timeout: int = 3
    flood_batch_size: int = 1000


class TelemetrySettings(BaseModel):
    """Settings for OpenTelelemtry and Sentry."""

    otel_enabled: bool = False
    otel_host: str | None = "jaeger-otel-collector.sri"
    otel_port: int | None = 4318
    otel_trace_endpoint: str | None = "/v1/traces"

    sentry_enabled: bool = False
    sentry_dsn: str | None = None
    traces_sample_rate: float = 0.1
    profiles_sample_rate: float = 1.0


class JobSettings(BaseModel):
    """Settings for job handling."""

    timeout: int = 300
    ttl: int = 2_592_000
    ttl_max: int = 31_536_000


class LogSettings(BaseModel):
    """Settings for log handling."""

    log_to_mongo: bool = True
    ttl: int = 1_209_600


def uppercase(value: str) -> str:
    """Make a string uppercase."""
    return value.upper()


class Neo4jSettings(BaseSettings):
    """Settings for the Tier 0 Neo4j interface."""

    query_timeout: int = 1600  # Time in seconds before a neo4j query should time out
    connect_retries: int = 25
    host: str = ""
    bolt_port: int = 7687
    username: str = ""
    password: str = ""


class Tier0Settings(BaseSettings):
    """Settings concern Tier 0 abstraction layers."""

    neo4j: Neo4jSettings = Neo4jSettings()


class GeneralConfig(BaseSettings):
    """General application config."""

    instance_env: str = "dev"

    log_level: Annotated[
        LogLevel,
        AfterValidator(uppercase),
    ] = "DEBUG"
    host: str = "0.0.0.0"
    port: int = 8080
    cors: CORSSettings = CORSSettings()
    workers: int | None = None  # Number of workers to use
    worker_concurrency: int = 10  # Number of concurrent jobs a worker may process
    allow_profiler: bool = True

    job: JobSettings = JobSettings()
    log: LogSettings = LogSettings()

    redis: RedisSettings = RedisSettings()
    mongo: MongoSettings = MongoSettings()
    tier0: Tier0Settings = Tier0Settings()
    telemetry: TelemetrySettings = TelemetrySettings()

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        case_sensitive=False,
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config/config.yaml",
        yaml_file_encoding="utf-8",
    )

    @classmethod
    @override
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Ensure proper setting priority order."""
        return (
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
            init_settings,
        )


CONFIG = GeneralConfig()
