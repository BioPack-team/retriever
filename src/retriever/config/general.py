import warnings
from typing import Annotated, ClassVar, override

from pydantic import AfterValidator, BaseModel, Field, SecretStr
from pydantic_file_secrets import FileSecretsSettingsSource, SettingsConfigDict
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

from retriever.types.general import LogLevel
from retriever.utils.general import CommentedSettings

# Filter warnings about secrets because they're optional
warnings.filterwarnings(
    action="ignore", message='directory "/run/secrets" does not exist'
)
warnings.filterwarnings(
    action="ignore", message='directory "config/secrets" does not exist'
)


class CORSSettings(BaseModel):
    """CORS-specific settings."""

    allow_origins: Annotated[
        list[str],
        Field(description="Origins allowed to make cross-origin requests."),
    ] = ["*"]
    allow_credentials: Annotated[
        bool, Field(description="Support cookies in cross-origin requests.")
    ] = True
    allow_methods: Annotated[
        list[str], Field(description="Methods allowed in cross-origin requests.")
    ] = ["*"]
    allow_headers: Annotated[
        list[str], Field(description="Headers allowed in cross-origin requests.")
    ] = ["*"]


class RedisSettings(BaseModel):
    """Redis client initialization settings."""

    host: str = "localhost"
    port: int = 6379
    password: SecretStr | None = None
    ssl_enabled: bool = False
    attempts: Annotated[
        int,
        Field(
            description="Number of attempts to accomplish operation before considering it failed."
        ),
    ] = 3
    timeout: Annotated[
        float, Field(description="Time before a redis operation is considered failed.")
    ] = 5
    shutdown_timeout: Annotated[
        int,
        Field(
            description="Time in seconds to wait for batched tasks to finish before force-quitting."
        ),
    ] = 3


class MongoSettings(BaseModel):
    """Mongodb client initialization settings."""

    host: str = "localhost"
    port: int = 27017
    username: str | None = None
    password: SecretStr | None = None
    authsource: str | None = None
    attempts: Annotated[
        int,
        Field(
            description="Number of attempts to accomplish operation before considering it failed."
        ),
    ] = 3
    shutdown_timeout: Annotated[
        int, Field(description="Time in seconds to wait for serialize task to finish.")
    ] = 3
    flood_batch_size: Annotated[
        int, Field(description=" Batch size for basic mongo inserts.")
    ] = 1000


class TelemetrySettings(BaseModel):
    """Settings for OpenTelelemtry and Sentry."""

    otel_enabled: bool = False
    otel_host: SecretStr | None = SecretStr("jaeger-otel-collector.sri")
    otel_port: int | None = 4318
    otel_trace_endpoint: str | None = "/v1/traces"

    sentry_enabled: bool = False
    sentry_dsn: SecretStr | None = None
    traces_sample_rate: Annotated[
        float, Field(description="Proportion of traces to send to Sentry.")
    ] = 0.1
    profiles_sample_rate: Annotated[
        float, Field(description="Proportion of sampled traces to profile.")
    ] = 1.0


class CallbackSettings(BaseModel):
    """Settings for callback handling."""

    retries: Annotated[
        int, Field(description="Number of times to retry the callback.")
    ] = 3
    timeout: Annotated[
        int,
        Field(description="Time in seconds before a callback attempt should time out."),
    ] = 10


class LookupSettings(BaseModel):
    """Settings pertaining to lookups."""

    timeout: Annotated[
        int,
        Field(
            description="Time in seconds before a job should time out, set to -1 to disable."
        ),
    ] = 10


class MetaKGSettings(BaseModel):
    """Settings pertaining to metakg queries."""

    timeout: Annotated[
        int,
        Field(
            description="Time in seconds before a job should time out, set to -1 to disable."
        ),
    ] = 10
    build_time: Annotated[
        int,
        Field(
            description="Time in seconds before MetaKG should be rebuilt. Set to -1 to only build at start."
        ),
    ] = -1


class JobSettings(BaseModel):
    """Settings for job handling."""

    callback: CallbackSettings = CallbackSettings()
    lookup: LookupSettings = LookupSettings()
    metakg: MetaKGSettings = MetaKGSettings()
    ttl: Annotated[
        int,
        Field(
            description="Time in seconds for job state to remain after it was last touched"
        ),
    ] = 2_592_000
    ttl_max: Annotated[
        int,
        Field(
            description="Time in seconds after which job state is cleared, regardless of last touch"
        ),
    ] = 31_536_000


class LogSettings(BaseModel):
    """Settings for log handling."""

    log_to_mongo: Annotated[bool, Field(description="Persist logs in MongoDB.")] = True
    ttl: Annotated[int, Field(description="Time in seconds for a log to persist.")] = (
        1_209_600
    )


def uppercase(value: str) -> str:
    """Make a string uppercase."""
    return value.upper()


class Neo4jSettings(BaseModel):
    """Settings for the Tier 0 Neo4j interface."""

    query_timeout: Annotated[
        int, Field(description="Time in seconds before a neo4j query should time out.")
    ] = 1600
    connect_retries: Annotated[
        int,
        Field(description="Number of retries before declaring a connection failure."),
    ] = 25
    host: str = ""
    bolt_port: int = 7687
    username: str = ""
    password: SecretStr = SecretStr("")
    database_name: str = "neo4j"


class Tier0Settings(BaseModel):
    """Settings concern Tier 0 abstraction layers."""

    neo4j: Neo4jSettings = Neo4jSettings()


class GeneralConfig(CommentedSettings):
    """General application config."""

    instance_env: Annotated[
        str,
        Field(
            description="Instance environment. Used in Sentry, userAgent of subqueries, instance-appropriate behavior, etc."
        ),
    ] = "dev"
    instance_idx: Annotated[
        int,
        Field(
            description="Instance index. Use when multiple Retriever instances are run, so a leader can be determined."
        ),
    ] = 0

    log_level: Annotated[
        LogLevel,
        AfterValidator(uppercase),
    ] = Field(
        default="DEBUG",
        description="Level of application logs to print/keep.",
    )
    host: Annotated[str, Field(description="Uvicorn listen host.")] = "0.0.0.0"
    port: Annotated[int, Field(description="Uvicorn listen port.")] = 8080
    cors: CORSSettings = CORSSettings()
    workers: Annotated[
        int | None, Field(description="Number of workers, defaults to n_cpus if unset.")
    ] = None
    allow_profiler: Annotated[
        bool,
        Field(
            description="Allow all queries to enable profiling with a query parameter."
        ),
    ] = True

    job: JobSettings = JobSettings()
    log: LogSettings = LogSettings()

    redis: RedisSettings = RedisSettings()
    mongo: MongoSettings = MongoSettings()
    tier0: Tier0Settings = Tier0Settings()
    telemetry: TelemetrySettings = TelemetrySettings()

    # Weird override happening here, see https://github.com/makukha/pydantic-file-secrets for an explanation
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(  # pyright:ignore[reportIncompatibleVariableOverride] This is the intended pattern
        case_sensitive=False,
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config/config.yaml",
        yaml_file_encoding="utf-8",
        secrets_dir=["config/secrets", "/run/secrets"],
        secrets_nested_delimiter="__",
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
            FileSecretsSettingsSource(file_secret_settings),
            file_secret_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            init_settings,
        )


CONFIG = GeneralConfig()
