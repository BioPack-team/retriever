from typing import Annotated, Any, ClassVar, override

import bmt
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, EmailStr
from pydantic.fields import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from retriever.utils.general import CommentedSettings

biolink = bmt.Toolkit()


class ContactInfo(BaseModel):
    """Contact information for the API maintainer."""

    email: EmailStr = "jcallaghan@scripps.edu"
    name: str = "Jackson Callaghan"
    url: str = "https://github.com/tokebe"
    x_id: Annotated[str, Field(serialization_alias="x-id")] = "tokebe"
    x_role: Annotated[str, Field(serialization_alias="x-role")] = (
        "responsible developer"
    )


class License(BaseModel):
    """License Information."""

    name: str = "Apache 2.0"
    url: str = "http://www.apache.org/licenses/LICENSE-2.0.html"


class Tag(BaseModel):
    """API metadata tags."""

    name: str
    description: str | None = None


class ExternalDocs(BaseModel):
    """Link to external documentation."""

    description: str
    url: str


class XTranslator(BaseModel):
    """Translator-specific metadata."""

    component: str = "KP"
    team: list[str] = ["DOGSURF"]
    biolink_version: str = "4.3.2"
    infores: Annotated[
        str,
        Field(description="Unique identifier for this component, used in provenance."),
    ] = "infores:retriever"
    externalDocs: ExternalDocs = ExternalDocs(
        description="The values for component and team are restricted according to this external JSON schema. See schema and examples at url.",
        url="https://github.com/NCATSTranslator/translator_extensions/blob/production/x-translator/",
    )


class TestDataLocation(BaseModel):
    """Link to testing data."""

    # TODO: add an appropriate test data location.
    url: str = "https://raw.githubusercontent.com/NCATS-Tangerine/translator-api-registry/master/biothings_explorer/sri-test-bte-ara.json"


class TestDataLocationObject(BaseModel):
    """Collection of testing data links."""

    default: TestDataLocation = TestDataLocation()


class XTrapi(BaseModel):
    """Trapi-specific metadata."""

    version: str = "1.6.0"
    multicuriesquery: Annotated[
        bool, Field(description="Supports advanced set interpretation.")
    ] = False
    pathfinderquery: Annotated[
        bool, Field(description="Supports Pathfinder-type queries.")
    ] = False
    asyncquery: Annotated[bool, Field(description="Supports async query type.")] = True
    operations: list[str] = ["lookup"]
    batch_size_limit: Annotated[
        int, Field(description="Maximum IDs on any one node.")
    ] = 300
    rate_limit: Annotated[
        int, Field(description="Maximum number of requests per minute.")
    ] = 300
    test_data_location: TestDataLocationObject = TestDataLocationObject()
    externalDocs: ExternalDocs = ExternalDocs(
        description="The values for version are restricted according to the regex in this external JSON schema. See schema and examples at url",
        url="https://github.com/NCATSTranslator/translator_extensions/blob/production/x-trapi/",
    )


class ResponseDescriptions(BaseModel):
    """Required response description fields."""

    meta_knowledge_graph: dict[str, str] = {
        "200": "Returns meta knowledge graph representation of this TRAPI web service.",
    }
    query: dict[str, str] = {
        "200": "OK. There may or may not be results. Note that some of the provided identifiers may not have been recognized.",
        "400": "Bad request. The request is invalid according to this OpenAPI schema OR a specific identifier is believed to be invalid somehow (not just unrecognized).",
        "413": "Payload too large. Indicates that batch size was over the limit specified in x-trapi.",
        "429": "Payload too large. Indicates that batch size was over the limit specified in x-trapi.",
    }
    asyncquery: dict[str, str] = {
        "200": "The query is accepted for processing and the Response will be sent to the callback url when complete."
    }
    asyncquery_status: dict[str, str] = {
        "200": "Returns the status and current logs of a previously submitted asyncquery.",
        "404": "job_id not found",
        "501": "Return code 501 indicates that this endpoint has not been implemented at this site. Sites that implement /asyncquery MUST implement /asyncquery_status/{job_id}, but those that do not implement /asyncquery SHOULD NOT implement /asyncquery_status.",
    }
    response: dict[str, str] = {
        "200": "Returns either the status and logs, or if complete, the complete response  of a previously submitted asyncquery."
    }
    logs: dict[str, str] = {
        "200": "Logs in either flat or structured JSON format.",
        "404": "Indicates this service is disabled by config.",
    }


class ServerConfig(CommentedSettings):
    """Configuration for a given server."""

    description: str
    url: str
    x_maturity: Annotated[
        str,
        Field(
            serialization_alias="x-maturity",
        ),
    ]


class OpenAPIConfig(CommentedSettings):
    """OpenAPI Metadata."""

    description: str = "Translator Knowledge Provider"
    version: str = "0.0.1"
    title: str = "Retriever"
    contact: ContactInfo = ContactInfo()
    license: License = License()
    termsOfService: str = "https://biothings.io/about"
    tags: list[Tag] = Field(
        default_factory=lambda: [
            Tag(name="meta_knowledge_graph"),
            Tag(name="query"),
            Tag(name="asyncquery"),
            Tag(name="asyncquery_status"),
            Tag(name="translator"),
            Tag(name="trapi"),
            Tag(name="biothings"),
        ]
    )
    # Have to set openapi_ prefix for any aliased fields for...some reason
    x_translator: Annotated[
        XTranslator,
        Field(
            serialization_alias="x-translator",
            description="Note that fields usually named with - instead use _ (e.g. x-translator -> x_translator).",
        ),
    ] = XTranslator()
    x_trapi: XTrapi = Field(
        default_factory=lambda: XTrapi(), serialization_alias="x-trapi"
    )
    response_descriptions: ResponseDescriptions = ResponseDescriptions()
    servers: list[ServerConfig] = Field(
        default_factory=lambda: [
            ServerConfig(
                description="DOGSURF AWS Dev instance",
                url="https://dev.retriever.biothings.io/",
                x_maturity="development",
            )
        ]
    )

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="openapi__",
        case_sensitive=False,
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config/openapi.yaml",
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


OPENAPI_CONFIG = OpenAPIConfig()


class TRAPI(FastAPI):
    """A TRAPI-spec FastAPI application."""

    @override
    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema:
            return self.openapi_schema

        # Build tags
        tags = [tag.model_dump(mode="json") for tag in OPENAPI_CONFIG.tags]
        if self.openapi_tags:
            tags.extend(self.openapi_tags)

        openapi_schema = get_openapi(
            description=OPENAPI_CONFIG.description,
            version=OPENAPI_CONFIG.version,
            title=OPENAPI_CONFIG.title,
            contact=OPENAPI_CONFIG.contact.model_dump(mode="json", by_alias=True),
            license_info=OPENAPI_CONFIG.license.model_dump(mode="json", by_alias=True),
            terms_of_service=OPENAPI_CONFIG.termsOfService,
            tags=tags,
            openapi_version=self.openapi_version,
            routes=self.routes,
        )

        openapi_schema["info"]["x-translator"] = OPENAPI_CONFIG.x_translator.model_dump(
            mode="json",
            by_alias=True,
        )
        openapi_schema["info"]["x-trapi"] = OPENAPI_CONFIG.x_trapi.model_dump(
            mode="json", by_alias=True
        )
        openapi_schema["servers"] = [
            server.model_dump(mode="json", by_alias=True)
            for server in OPENAPI_CONFIG.servers
        ]

        self.openapi_schema: dict[str, Any] | None = openapi_schema
        return self.openapi_schema
