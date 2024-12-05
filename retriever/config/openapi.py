from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, BaseSettings, EmailStr
from pydantic.env_settings import SettingsSourceCallable
from pydantic.fields import Field

from .util import get_yaml_settings


class ContactInfo(BaseModel):
    email: EmailStr = EmailStr("jcallaghan@scripps.edu")
    name: str = "Jackson Callaghan"
    url: str = "https://github.com/tokebe"
    x_id: str = Field(default="tokebe")
    x_role: str = Field(default="responsible developer")

    # class Config:
    #     alias_generator = lambda string: string.replace("_", "-")


class License(BaseModel):
    name: str = "Apache 2.0"
    url: str = "http://www.apache.org/licenses/LICENSE-2.0.html"


class Tag(BaseModel):
    name: str
    description: Optional[str] = None


class ExternalDocs(BaseModel):
    description: str
    url: str


class XTranslator(BaseModel):
    component: str = ""
    team: list[str] = ["biopack"]
    biolink_version: str = Field(default="4.2.1")
    infores: str = "retriever"
    externalDocs: ExternalDocs = ExternalDocs(
        description="The values for component and team are restricted according to this external JSON schema. See schema and examples at url",
        url="https://github.com/NCATSTranslator/translator_extensions/blob/production/x-translator/",
    )


class TestDataLocation(BaseModel):
    url: str


class TestDataLocationObject(BaseModel):
    default: TestDataLocation


class XTrapi(BaseModel):
    version: str = "1.5.0"
    multicuriesquery: bool = False
    pathfinderquery: bool = False
    asyncquery: bool = True
    operations: list[str] = ["lookup"]
    batch_size_limit: int = 300
    rate_limit: int = 300
    test_data_location: TestDataLocationObject = TestDataLocationObject(
        default=TestDataLocation(
            url="https://raw.githubusercontent.com/NCATS-Tangerine/translator-api-registry/master/biothings_explorer/sri-test-bte-ara.json"
        )
    )
    externalDocs: ExternalDocs = ExternalDocs(
        description="The values for version are restricted according to the regex in this external JSON schema. See schema and examples at url",
        url="https://github.com/NCATSTranslator/translator_extensions/blob/production/x-trapi/",
    )


class OpenAPISettings(BaseSettings):
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

    class Config:
        env_prefix = "openapi__"
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
                get_yaml_settings(Path(__file__).parent / "openapi.yaml"),
                init_settings,
                file_secret_settings,
            )


class TRAPI(FastAPI):
    def openapi(self) -> dict[str, Any]:
        if self.openapi_schema:
            return self.openapi_schema

        openapi_settings = OpenAPISettings()

        # Build tags
        tags = [tag.dict() for tag in openapi_settings.tags]
        if self.openapi_tags:
            tags.extend(self.openapi_tags)

        openapi_schema = get_openapi(
            description=openapi_settings.description,
            version=openapi_settings.version,
            title=openapi_settings.title,
            contact={
                name.replace("_", "-"): value
                for name, value in openapi_settings.contact.dict().items()
            },
            license_info=openapi_settings.license.dict(),
            terms_of_service=openapi_settings.termsOfService,
            tags=tags,
            openapi_version=self.openapi_version,
            routes=self.routes,
        )

        openapi_schema["info"]["x-translator"] = openapi_settings.x_translator.dict(
            by_alias=True
        )
        openapi_schema["info"]["x-trapi"] = openapi_settings.x_trapi.dict(by_alias=True)

        self.openapi_schema = openapi_schema
        return self.openapi_schema
