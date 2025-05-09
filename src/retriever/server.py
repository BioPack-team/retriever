import functools
import io
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, time, timedelta
from typing import Annotated, Any, Literal

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi import Response as StandardResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from retriever.config.general import CONFIG
from retriever.config.logger import configure_logging
from retriever.config.openapi import OPENAPI_CONFIG, TRAPI

# from retriever.tasks.task_queue import get_job_state, make_query, retriever_queue
from retriever.tasks.query import get_job_state, make_query
from retriever.type_defs import ErrorDetail, LogLevel
from retriever.utils.exception_handlers import ensure_cors
from retriever.utils.logs import (
    add_mongo_sink,
    cleanup,
    objs_to_json,
    structured_log_to_trapi,
)

# from retriever.utils.mongo import MongoClient, MongoQueue
from retriever.utils.mongo import MONGO_CLIENT, MONGO_QUEUE
from retriever.utils.redis import test_redis
from retriever.utils.telemetry import configure_telemetry


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Lifespan hook for any setup/shutdown behavior."""
    # Startup
    configure_logging()
    await MONGO_CLIENT.initialize()
    await MONGO_QUEUE.start_process_task()
    # retriever_queue.mongo_queue = MONGO_QUEUE.queue
    add_mongo_sink()
    await test_redis()

    # workers = start_workers(MONGO_QUEUE.queue)

    yield  # Separates startup/shutdown phase

    # Shutdown
    # stop_workers(workers)
    # await retriever_queue.disconnect()
    await MONGO_QUEUE.stop_process_task()
    await MONGO_CLIENT.close()
    await cleanup()


app = TRAPI(lifespan=lifespan)

# Set up CORS
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


# Add a yaml endpoint, for completeness' sake
@app.get("/openapi.yaml", include_in_schema=False)
@functools.lru_cache
def openapi_yaml() -> StandardResponse:
    """Retreive the OpenAPI specs in yaml format."""
    openapi_json = app.openapi()
    yaml_str = io.StringIO()
    yaml.dump(openapi_json, yaml_str)
    return StandardResponse(yaml_str.getvalue(), media_type="text/yaml")


# TODO: implement by-smartapi, possibly just by a query value
# Or maybe an infores path param?
@app.get(
    "/meta_knowledge_graph",
    tags=["meta_knowledge_graph"],
    response_description=OPENAPI_CONFIG.response_descriptions.meta_knowledge_graph.get(
        "200", ""
    ),
)
async def meta_knowledge_graph(request: Request, response: Response) -> dict[str, Any]:
    """Retrieve the Meta-Knowledge Graph."""
    return await make_query("metakg", request, response)
    # return {"logs": list(logs)}


@app.post(
    "/query",
    tags=["query"],
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
# TODO: replace body type with updated reasoner-pydantic types
async def query(
    request: Request, response: Response, body: dict[str, Any]
) -> dict[str, Any]:
    """Initiate a synchronous query."""
    return await make_query("lookup", request, response, body)
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
    body: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Initiate an asynchronous query."""
    return await make_query(
        "lookup", request, response, body, background_tasks=background_tasks
    )


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
) -> dict[str, Any]:
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

    return job_dict


@app.get(
    "/asyncquery_response/{job_id}",
    tags=["asyncquery_status"],
    response_description=OPENAPI_CONFIG.response_descriptions.asyncquery_response.get(
        "200", ""
    ),
    responses={
        404: {
            "description": OPENAPI_CONFIG.response_descriptions.asyncquery_status.get(
                "404", ""
            ),
            "model": str,
        },
    },
)
async def asyncquery_response(
    request: Request, response: Response, job_id: str
) -> dict[str, Any]:
    """Get the response of an asynchronous query."""
    return await get_job_state(job_id, request, response)


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
            description="Start datetime to search logs. Defaults to yesterday at midnight if `all_dates` is not set.",
        ),
    ] = None,
    end: Annotated[
        datetime | None,
        Query(
            description="End datetime to search logs. Defaults to present time.",
        ),
    ] = None,
    level: LogLevel = "DEBUG",
    all_dates: Annotated[
        bool, Query(description="Retrieve all logs without filtering by date.")
    ] = False,
    job_id: str | None = None,
    fmt: Annotated[
        Literal["flat", "trapi", "default"],
        Query(
            description="Respond with a specific format. flat: plaintext log lines; trapi: TRAPI-style logs; default: default structured format"
        ),
    ] = "default",
) -> StreamingResponse:
    """Retrieve MongoDB-saved server logs."""
    if not CONFIG.log.log_to_mongo:
        raise HTTPException(404, detail="Persisted logging not enabled.")

    if (
        not start and not job_id and not all_dates
    ):  # Get all logs since midnight yesterday
        start = datetime.combine(datetime.today(), time.min) - timedelta(days=1)

    if fmt == "flat":
        logs = MONGO_CLIENT.get_flat_logs(start, end, level, job_id)
    else:
        logs = MONGO_CLIENT.get_logs(start, end, level, job_id)
        if fmt == "trapi":
            logs = structured_log_to_trapi(logs)
        logs = objs_to_json(logs)

    return StreamingResponse(
        logs,
        media_type="application/json" if fmt != "flat" else "text/plain",
    )
