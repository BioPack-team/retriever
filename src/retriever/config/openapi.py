from typing import Any, ClassVar, override

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


class ContactInfo(BaseModel):
    """Contact information for the API maintainer."""

    email: EmailStr = "jcallaghan@scripps.edu"
    name: str = "Jackson Callaghan"
    url: str = "https://github.com/tokebe"
    x_id: str = "tokebe"
    x_role: str = "responsible developer"


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

    component: str = ""
    team: list[str] = ["biopack"]
    biolink_version: str = "4.2.1"
    infores: str = "retriever"
    externalDocs: ExternalDocs = ExternalDocs(
        description="The values for component and team are restricted according to this external JSON schema. See schema and examples at url.",
        url="https://github.com/NCATSTranslator/translator_extensions/blob/production/x-translator/",
    )


class TestDataLocation(BaseModel):
    """Link to testing data."""

    url: str


class TestDataLocationObject(BaseModel):
    """Collection of testing data links."""

    default: TestDataLocation


class XTrapi(BaseModel):
    """Trapi-specific metadata."""

    version: str = "1.5.0"
    multicuriesquery: bool = False
    pathfinderquery: bool = False
    asyncquery: bool = True
    operations: list[str] = ["lookup"]
    batch_size_limit: int = 300
    rate_limit: int = 300
    test_data_location: TestDataLocationObject = TestDataLocationObject(
        default=TestDataLocation(
            url="https://raw.githubusercontent.com/NCATS-Tangerine/translator-api-registry/master/biothings_explorer/sri-test-bte-ara.json",
        ),
    )
    externalDocs: ExternalDocs = ExternalDocs(
        description="The values for version are restricted according to the regex in this external JSON schema. See schema and examples at url",
        url="https://github.com/NCATSTranslator/translator_extensions/blob/production/x-trapi/",
    )


class ResponseDescriptions(BaseModel):
    """Required response description fields."""

    meta_knowledge_graph: dict[str, str]
    query: dict[str, str]
    asyncquery: dict[str, str]
    asyncquery_status: dict[str, str]
    asyncquery_response: dict[str, str]
    logs: dict[str, str]


class OpenAPIConfig(BaseSettings):
    """OpenAPI Metadata."""

    description: str = "Translator Knowledge Provider"
    version: str = "0.0.1"
    title: str = "Retriever"
    contact: ContactInfo = ContactInfo()
    license: License = License()
    termsOfService: str = "https://biothings.io/about"
    tags: list[Tag] = [
        Tag(name="meta_knowledge_graph"),
        Tag(name="query"),
        Tag(name="asyncquery"),
        Tag(name="asyncquery_status"),
        Tag(name="translator"),
        Tag(name="trapi"),
        Tag(name="biothings"),
    ]
    # Have to set openapi_ prefix for any aliased fields for...some reason
    x_translator: XTranslator = Field(default=XTranslator())
    x_trapi: XTrapi = Field(default=XTrapi())
    response_descriptions: ResponseDescriptions

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
            contact={
                name.replace("_", "-"): value
                for name, value in OPENAPI_CONFIG.contact.model_dump(
                    mode="json"
                ).items()
            },
            license_info=OPENAPI_CONFIG.license.model_dump(mode="json"),
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

        self.openapi_schema: dict[str, Any] | None = openapi_schema
        return self.openapi_schema
