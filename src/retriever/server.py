import functools
import io
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Annotated, Literal

import yaml
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from reasoner_pydantic import AsyncQueryStatusResponse as TRAPIAsyncQueryStatusResponse

from retriever.config.general import CONFIG
from retriever.config.logger import configure_logging
from retriever.config.openapi import OPENAPI_CONFIG, TRAPI
from retriever.data_tiers import tier_manager
from retriever.lookup.subclass import SubclassMapping
from retriever.lookup.subquery import SubqueryDispatcher
from retriever.lookup.utils import QueryDumper
from retriever.metadata.optable import OpTableManager
from retriever.query import get_job_state, make_query
from retriever.types.general import APIInfo, ErrorDetail, LogLevel
from retriever.types.trapi_pydantic import AsyncQuery as TRAPIAsyncQuery
from retriever.types.trapi_pydantic import Query as TRAPIQuery
from retriever.types.trapi_pydantic import Response as TRAPIResponse
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils.examples import EXAMPLE_QUERY
from retriever.utils.exception_handlers import ensure_cors
from retriever.utils.logs import (
    add_mongo_sink,
    cleanup,
    objs_to_json,
    structured_log_to_trapi,
)
from retriever.utils.mongo import MongoClient, MongoQueue
from retriever.utils.redis import RedisClient
from retriever.utils.telemetry import configure_telemetry

configure_logging()


# Lifespan handling for each FastAPI worker (not main process, see __main__.py)
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Lifespan hook for any setup/shutdown behavior."""
    # Startup
    os.environ["PYTHONHASHSEED"] = "0"  # So reasoner_pydantic hashing is deterministic
    await MongoClient().initialize()
    await MongoQueue().initialize()
    add_mongo_sink()
    await RedisClient().initialize()
    await OpTableManager().initialize()
    await tier_manager.connect_drivers()
    await SubclassMapping().initialize()
    query_dumper = QueryDumper()
    if CONFIG.tier0.dump_queries or CONFIG.tier1.dump_queries:
        await query_dumper.initialize()
    await SubqueryDispatcher().initialize()

    yield  # Separates startup/shutdown phase

    # Shutdown
    await SubqueryDispatcher().wrapup()
    if query_dumper.initialized:
        await query_dumper.wrapup()
    await SubclassMapping().wrapup()
    await tier_manager.close_drivers()
    await OpTableManager().wrapup()
    await RedisClient().wrapup()
    await MongoQueue().wrapup()
    await MongoClient().close()
    await cleanup()


app = TRAPI(lifespan=lifespan)

# Set up CORS / exception handling
app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.cors.allow_origins,
    allow_credentials=CONFIG.cors.allow_credentials,
    allow_methods=CONFIG.cors.allow_methods,
    allow_headers=CONFIG.cors.allow_headers,
)


@app.exception_handler(500)
async def exception_ensure_cors(request: Request, exc: Exception) -> Response:
    """Ensure CORS is not lost on exception."""
    return await ensure_cors(app, request, exc)


# Set up Sentry and Otel
configure_telemetry(app)

# Configure profiling middleware
if CONFIG.allow_profiler:
    from pyinstrument import Profiler
    from pyinstrument.renderers import SpeedscopeRenderer

    @app.middleware("http")
    async def profile_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Profile the api call when query parameter `profile` is set to True."""
        profile = request.query_params.get("profile", "false")
        if profile and profile != "false":
            profiler = Profiler(interval=0.01, async_mode="enabled")
            profiler.start()
            await call_next(request)
            profiler.stop()
            speedscope_results = profiler.output(renderer=SpeedscopeRenderer())
            return Response(content=speedscope_results, media_type="application/json")
        return await call_next(request)


# Add a yaml endpoint, for completeness' sake
@app.get("/openapi.yaml", include_in_schema=False)
@functools.lru_cache
def openapi_yaml() -> Response:
    """Retreive the OpenAPI specs in yaml format."""
    openapi_json = app.openapi()
    yaml_str = io.StringIO()
    yaml.dump(openapi_json, yaml_str)
    return Response(yaml_str.getvalue(), media_type="text/yaml")


@app.get(
    "/meta_knowledge_graph",
    tags=["meta_knowledge_graph"],
    response_description=OPENAPI_CONFIG.response_descriptions.meta_knowledge_graph.get(
        "200", ""
    ),
)
async def meta_knowledge_graph(
    request: Request,
    response: Response,
    tier: Annotated[
        list[TierNumber],
        Query(
            description="Data Tier to use. Set multiple times to access multiple Tiers simultaneously.",
            default_factory=lambda: [0],
        ),
    ],
) -> ORJSONResponse:
    """Retrieve the Meta-Knowledge Graph."""
    response_dict = await make_query("metakg", APIInfo(request, response), tiers=tier)
    return ORJSONResponse(response_dict)
    # return {"logs": list(logs)}


@app.get(
    "/metadata/tier_{tier}",
    tags=["metadata"],
    response_description=OPENAPI_CONFIG.response_descriptions.metadata.get("200", ""),
)
async def metadata(
    request: Request, response: Response, tier: TierNumber
) -> ORJSONResponse:
    """Retrieve the metadata associated with a given Data Tier."""
    response_dict = await make_query(
        func="metadata", ctx=APIInfo(request, response), tiers=[tier]
    )
    return ORJSONResponse(response_dict)


@app.post(
    "/query",
    tags=["query"],
    response_model=TRAPIResponse,
    response_description=OPENAPI_CONFIG.response_descriptions.query.get("200", ""),
    responses={
        400: {
            "description": OPENAPI_CONFIG.response_descriptions.query.get("400", ""),
            "model": str,
        },
        413: {
            "description": OPENAPI_CONFIG.response_descriptions.query.get("413", ""),
            "model": str,
        },
        429: {
            "description": OPENAPI_CONFIG.response_descriptions.query.get("429", ""),
            "model": str,
        },
        500: {
            "description": OPENAPI_CONFIG.response_descriptions.query.get("500", ""),
            "model": str,
        },
    },
)
async def query(
    request: Request,
    response: Response,
    body: Annotated[TRAPIQuery, Body(examples=[EXAMPLE_QUERY])],
) -> ORJSONResponse:
    """Initiate a synchronous query."""
    response_dict = await make_query("lookup", APIInfo(request, response), body=body)
    return ORJSONResponse(response_dict)
    # return {}


@app.post(
    "/asyncquery",
    tags=["asyncquery"],
    response_description=OPENAPI_CONFIG.response_descriptions.asyncquery.get("200", ""),
    responses={
        400: {
            "description": OPENAPI_CONFIG.response_descriptions.query.get("400", ""),
            "model": str,
        },
        413: {
            "description": OPENAPI_CONFIG.response_descriptions.query.get("413", ""),
            "model": str,
        },
        429: {
            "description": OPENAPI_CONFIG.response_descriptions.query.get("429", ""),
            "model": str,
        },
        500: {
            "description": OPENAPI_CONFIG.response_descriptions.query.get("500", ""),
            "model": str,
        },
    },
)
async def asyncquery(
    request: Request,
    response: Response,
    body: TRAPIAsyncQuery,
    background_tasks: BackgroundTasks,
) -> ORJSONResponse:
    """Initiate an asynchronous query."""
    response_dict = await make_query(
        "lookup", APIInfo(request, response, background_tasks), body=body
    )
    return ORJSONResponse(response_dict)


@app.get(
    "/asyncquery_status/{job_id}",
    tags=["asyncquery_status"],
    response_description=OPENAPI_CONFIG.response_descriptions.asyncquery_status.get(
        "200", ""
    ),
    responses={
        404: {
            "description": OPENAPI_CONFIG.response_descriptions.asyncquery_status.get(
                "404", ""
            ),
            "model": str,
        },
        501: {
            "description": OPENAPI_CONFIG.response_descriptions.asyncquery_status.get(
                "501", ""
            )
        },
    },
)
async def asyncquery_status(
    request: Request, response: Response, job_id: str
) -> ORJSONResponse:
    """Get the status of an asynchronous query."""
    job_dict = await get_job_state(job_id, request, response)

    # Remove keys not meant for asyncquery_status
    del_keys: list[str] = [
        key
        for key in job_dict
        if key
        not in (
            "job_id",
            "status",
            "description",
            "logs",
            "response_url",
            "error",
            "trace",
        )
    ]
    for key in del_keys:
        del job_dict[key]

    return ORJSONResponse(job_dict)


@app.get(
    "/response/{job_id}",
    tags=["asyncquery_status"],
    response_model=TRAPIResponse | TRAPIAsyncQueryStatusResponse,
    response_description=OPENAPI_CONFIG.response_descriptions.response.get("200", ""),
    responses={
        404: {
            "description": OPENAPI_CONFIG.response_descriptions.asyncquery_status.get(
                "404", ""
            ),
            "model": str,
        },
    },
)
async def response(request: Request, response: Response, job_id: str) -> ORJSONResponse:
    """Get the response of an asynchronous query."""
    job_dict = await get_job_state(job_id, request, response)
    return ORJSONResponse(job_dict)


@app.get(
    "/logs",
    tags=["logs"],
    response_description=OPENAPI_CONFIG.response_descriptions.logs.get("200", ""),
    responses={
        404: {
            "description": OPENAPI_CONFIG.response_descriptions.logs.get("404", ""),
            "model": ErrorDetail,
        }
    },
)
async def logs(  # noqa: PLR0913 Can't reduce args due to FastAPI endpoint format
    start: Annotated[
        datetime | None,
        Query(
            description="Start datetime to search logs. Defaults to last hour if `lookback` is not set.",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        Query(
            description="End datetime to search logs. Defaults to present time.",
        ),
    ] = None,
    level: LogLevel = "DEBUG",
    lookback: Annotated[
        float | None,
        Query(
            description="Number of hours back from present time to retrieve. Overrides `start`."
        ),
    ] = None,
    job_id: Annotated[
        str | None,
        Query(
            description="ID of a previously-run job to search for. Limits logs to those related to that job."
        ),
    ] = None,
    fmt: Annotated[
        Literal["flat", "trapi", "struct"],
        Query(
            description="Respond with a specific format. flat: plaintext log lines; trapi: TRAPI-style logs; struct: loguru-structured format"
        ),
    ] = "flat",
) -> StreamingResponse:
    """Retrieve MongoDB-saved server logs."""
    if not CONFIG.log.log_to_mongo:
        raise HTTPException(404, detail="Persisted logging not enabled.")

    if lookback is not None:
        start = datetime.now() - timedelta(seconds=lookback * 60 * 60)

    elif not start and not job_id:  # Get all logs from last hour
        start = datetime.now() - timedelta(hours=1)

    if fmt == "flat":
        logs = MongoClient().get_flat_logs(start, end, level, job_id)
    else:
        logs = MongoClient().get_logs(start, end, level, job_id)
        if fmt == "trapi":
            logs = structured_log_to_trapi(logs)
        use_jsonl = fmt == "struct"
        logs = objs_to_json(logs, jsonl=use_jsonl)

    return StreamingResponse(
        logs,
        media_type="application/json" if fmt != "flat" else "text/plain",
    )
