from typing import Any
from fastapi import FastAPI, Request
import json
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from ratelimit.types import Scope
from reasoner_pydantic import Query, Response
from fastapi import Response as StandardResponse

from retriever.config.openapi import TRAPI
from retriever.config.cors import CorsSettings
from retriever.config.config import GeneralConfig
import functools
import yaml
import io
from ratelimit import RateLimitMiddleware, Rule
from ratelimit.backends.simple import MemoryBackend
from pprint import pprint

config = GeneralConfig()

APP = TRAPI()
APP.add_middleware(
    CORSMiddleware,
    **CorsSettings().dict(),
)


async def AUTH_FUNCTION(scope: Scope) -> tuple[str, str]:
    user_agent = next(
        (value for name, value in scope.get("headers", []) if name == b"user-agent"),
        b"",
    ).decode()
    print(f"User agent is {user_agent}")
    if "shepherd" in user_agent:
        return "shepherd", "shepherd"
    else:
        return "other", "other"


APP.add_middleware(
    RateLimitMiddleware,
    authenticate=AUTH_FUNCTION,
    backend=MemoryBackend(),
    config={
        r".*": [
            Rule(group="shepherd", minute=config.rate_limit.special),
            Rule(group="other", minute=config.rate_limit.general),
        ]
    },
)


# Add a yaml endpoint, for completeness' sake
@APP.get("/openapi.yaml", include_in_schema=False)
@functools.lru_cache()
def openapi_yaml(request: Request) -> StandardResponse:
    openapi_json = APP.openapi()
    yaml_str = io.StringIO()
    yaml.dump(openapi_json, yaml_str)
    return StandardResponse(yaml_str.getvalue(), media_type="text/yaml")


# TODO: implement by-smartapi, possibly just by a query value
# Or maybe an infores path param?
@APP.get("/meta_knowledge_graph")
async def meta_knowledge_graph(request: Request) -> dict[str, Any]:
    """Retrieve the Meta-Knowledge Graph"""
    # TODO: implement
    return {}


@APP.post("/query")
async def query(request: Request, body: Query) -> dict[str, Any]:
    """Initiate a synchronous query"""
    # TODO: implement
    return body.dict()


@APP.post("/asyncquery")
async def asyncquery(request: Request, body: Query) -> dict[str, Any]:
    """Initiate an asynchronous query"""
    # TODO: implement
    return body.dict()


@APP.get("/asyncquery_status/{job_id}")
async def asyncquery_status(request: Request) -> dict[str, Any]:
    """Get the status of an asynchronous query"""
    # TODO: implement
    return {}


@APP.get("/asyncquery_response/{job_id}")
async def asyncquery_response(request: Request) -> dict[str, Any]:
    """Get the response of an asynchronous query"""
    # TODO: implement
    return {}
