import warnings
from pathlib import Path
from typing import Annotated, ClassVar, override

from pydantic import AfterValidator, BaseModel, Field, FilePath, SecretStr
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


class ElasticSearchSettings(BaseModel):
    """Settings for the Tier 1 ElasticSearch interface."""

    query_timeout: Annotated[
        int,
        Field(
            description="Time in seconds before a Elasticsearch query should time out."
        ),
    ] = 1600
    connect_retries: Annotated[
        int,
        Field(description="Number of retries before declaring a connection failure."),
    ] = 5
    host: str = ""
    port: int = 9200
    database_name: str = "elasticsearch"

    # use merged edges index
    index_name: str = "rtx_kg2"


class Tier1Settings(BaseModel):
    """Settings concern Tier 1 abstraction layers."""

    backend: str = "elasticsearch"
    metakg_file: FilePath = Path("data/rtx-kg2-metakg.json")
    backend_infores: str = "infores:rtx-kg2"
    elasticsearch: ElasticSearchSettings = ElasticSearchSettings()


class Neo4jSettings(BaseModel):
    """Settings for the Tier 0 Neo4j interface."""

    query_timeout: Annotated[
        int, Field(description="Time in seconds before a neo4j query should time out.")
    ] = 1600
    connect_retries: Annotated[
        int,
        Field(description="Number of retries before declaring a connection failure."),
    ] = 5
    host: str = ""
    bolt_port: int = 7687
    username: str = ""
    password: SecretStr = SecretStr("")
    database_name: str = "neo4j"


class DgraphSettings(BaseModel):
    """Settings for the Tier 0 Dgraph interface."""

    host: str = "localhost"
    http_port: int = 8080
    grpc_port: int = 9080
    use_tls: bool = False
    query_timeout: Annotated[
        int,
        Field(
            description="Time in seconds before a Dgraph query should time out.",
        ),
    ] = 60
    connect_retries: Annotated[
        int,
        Field(
            description="Number of retries before declaring a connection failure.",
        ),
    ] = 5
    grpc_max_send_message_length: Annotated[
        int,
        Field(
            description="gRPC max send message length in bytes (-1 for unlimited).",
        ),
    ] = -1
    grpc_max_receive_message_length: Annotated[
        int,
        Field(
            description="gRPC max receive message length in bytes (-1 for unlimited).",
        ),
    ] = -1

    @property
    def http_endpoint(self) -> str:
        """Get the complete HTTP endpoint URL for Dgraph HTTP API."""
        scheme = "https" if self.use_tls else "http"
        return f"{scheme}://{self.host}:{self.http_port}"

    @property
    def grpc_endpoint(self) -> str:
        """Get the complete gRPC endpoint URL for Dgraph gRPC API."""
        return f"{self.host}:{self.grpc_port}"


class Tier0Settings(BaseModel):
    """Settings concern Tier 0 abstraction layers."""

    backend: str = "dgraph"
    metakg_file: FilePath = Path("data/rtx-kg2-metakg.json")
    backend_infores: str = "infores:automat-robokop"
    neo4j: Neo4jSettings = Neo4jSettings()
    dgraph: DgraphSettings = DgraphSettings()


class GeneralConfig(CommentedSettings):
    """General application config."""

    debug: Annotated[
        bool,
        Field(
            description="Run server with increased compatibility for breakpoint debugging."
        ),
    ] = False
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
    trust_proxy: Annotated[
        bool, Field(description="Use proxy IP headers (such as in nginx cases)")
    ] = True
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
    category_conflations: list[set[str]] = Field(
        description="A list of category conflation sets where a QNode having one member in the set will inheret the others",
        default_factory=lambda: [
            {"biolink:Gene", "biolink:Protein"},
            {"biolink:Drug", "biolink:ChemicalEntity"},
        ],
    )

    job: JobSettings = JobSettings()
    log: LogSettings = LogSettings()

    redis: RedisSettings = RedisSettings()
    mongo: MongoSettings = MongoSettings()
    tier0: Tier0Settings = Tier0Settings()
    tier1: Tier1Settings = Tier1Settings()
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
