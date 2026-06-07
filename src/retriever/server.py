import functools
import io
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from http import HTTPStatus
from typing import Annotated, Any, Literal

import yaml
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    ORJSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from loguru import logger
from reasoner_pydantic import AsyncQueryStatusResponse as TRAPIAsyncQueryStatusResponse

from retriever.config.general import CONFIG
from retriever.config.logger import configure_logging
from retriever.config.openapi import OPENAPI_CONFIG, TRAPI
from retriever.data_tiers import tier_manager
from retriever.lookup.subclass import SubclassMapping
from retriever.lookup.subquery import SubqueryDispatcher
from retriever.lookup.utils import QueryDumper
from retriever.metadata.optable import OpTableManager
from retriever.query import (
    get_job_response,
    get_job_status,
    make_lookup_query,
    make_metadata_query,
    make_metakg_query,
)
from retriever.types.general import APIInfo, ErrorDetail, LogLevel
from retriever.types.trapi_pydantic import AsyncQuery as TRAPIAsyncQuery
from retriever.types.trapi_pydantic import Query as TRAPIQuery
from retriever.types.trapi_pydantic import Response as TRAPIResponse
from retriever.types.trapi_pydantic import TierNumber
from retriever.utils import service_health, worker
from retriever.utils.examples import EXAMPLE_QUERY
from retriever.utils.exception_handlers import ensure_cors, http_exception_handler
from retriever.utils.general import tolerate_init
from retriever.utils.logs import (
    add_mongo_sink,
    cleanup,
    objs_to_json,
    structured_log_to_trapi,
)
from retriever.utils.mongo import MongoClient, MongoQueue
from retriever.utils.redis import RedisClient
from retriever.utils.telemetry import configure_telemetry
from retriever.utils.trapi import append_aggregator_source
from retriever.utils.version import get_version

configure_logging()

JOB_ID_PATTERN = r"^[a-zA-Z0-9]+$"


# Lifespan handling for each FastAPI worker (not main process, see __main__.py)
async def _refresh_worker_registration(pid: int, started_at: datetime) -> None:
    """One heartbeat tick: re-register the worker (and main, in debug)."""
    await RedisClient().register_worker(
        pid, started_at, CONFIG.redis.process_ttl_seconds
    )
    if CONFIG.debug:
        # In debug mode this worker is also the main entry-point.
        await RedisClient().register_main(
            pid, started_at, CONFIG.redis.process_ttl_seconds
        )


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Lifespan hook for any setup/shutdown behavior."""
    # Startup
    os.environ["PYTHONHASHSEED"] = "0"  # So reasoner_pydantic hashing is deterministic
    await MongoClient().initialize()
    await MongoQueue().initialize()
    add_mongo_sink()
    await RedisClient().initialize()

    worker_pid = os.getpid()
    worker_started_at = datetime.now().astimezone()
    worker.init(worker_started_at)
    await tolerate_init(
        "Worker registration",
        RedisClient().register_worker(
            worker_pid, worker_started_at, CONFIG.redis.process_ttl_seconds
        ),
    )
    RedisClient().start_heartbeat(
        lambda: _refresh_worker_registration(worker_pid, worker_started_at),
        role_label="Worker",
    )
    # In debug mode main and worker are the same.
    # Register it under both so /status.processes.main isn't
    # null. In production main is registered by __main__.py.
    if CONFIG.debug:
        await tolerate_init(
            "Main registration",
            RedisClient().register_main(
                worker_pid, worker_started_at, CONFIG.redis.process_ttl_seconds
            ),
        )

    await tolerate_init("OpTable pull", OpTableManager().initialize())
    await tier_manager.initialize_drivers()
    await tolerate_init("Subclass map pull", SubclassMapping().initialize())
    query_dumper = QueryDumper()
    if CONFIG.tier0.dump_queries or CONFIG.tier1.dump_queries:
        await query_dumper.initialize()
    await SubqueryDispatcher().initialize()

    yield  # Separates startup/shutdown phase

    # Shutdown
    try:
        await RedisClient().unregister_worker(worker_pid)
    except Exception:
        logger.warning(
            "Worker deregistration failed; entry will expire via TTL.",
            no_mongo_log=True,
        )
    await SubqueryDispatcher().wrapup()
    if query_dumper.initialized:
        await query_dumper.wrapup()
    await SubclassMapping().wrapup()
    await tier_manager.wrapup_drivers()
    await OpTableManager().wrapup()
    await RedisClient().wrapup()
    await MongoQueue().wrapup()
    await MongoClient().wrapup()
    await cleanup()


app = TRAPI(lifespan=lifespan)

# Mount the /status/* dashboard API (route handlers live in retriever.status).
from retriever.status import router as status_router  # noqa: E402

app.include_router(status_router)

# Mount the dashboard at /monitor - static HTML/JS/CSS that polls the
# /status/* endpoints from the browser. No server-side session state.
from pathlib import Path  # noqa: E402

from fastapi.staticfiles import StaticFiles  # noqa: E402

_MONITOR_STATIC = Path(__file__).parent / "monitor" / "static"
app.mount(
    "/monitor",
    StaticFiles(directory=_MONITOR_STATIC, html=True),
    name="monitor",
)

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


app.add_exception_handler(HTTPException, http_exception_handler)  # pyright: ignore[reportArgumentType]


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


@app.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    """Redirect / to /docs."""
    return RedirectResponse(url="/docs")


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
        TierNumber | None,
        Query(
            description="Data Tier to use. Leave unset to view all.",
        ),
    ] = None,
) -> ORJSONResponse:
    """Retrieve the Meta-Knowledge Graph."""
    snap = service_health.Snapshot()
    if tier is not None:
        resolved = snap.select_tier(None, tier, allow_fallback=False)
        if isinstance(resolved[0], HTTPStatus):
            status_code, detail = resolved
            raise HTTPException(status_code, detail=detail)
    elif snap.http_status_for("/meta_knowledge_graph") is not None:
        tier_statuses = [tier_manager.get_driver(n).status() for n in range(2)]
        raise HTTPException(
            HTTPStatus.FAILED_DEPENDENCY,
            detail=ErrorDetail(
                detail="No tier backends are available.",
                additional_info={
                    f"tier_{n}": {
                        "outage_time": s["last_outage"],
                        "outage_error": s["error"],
                    }
                    for n, s in enumerate(tier_statuses)
                },
            ),
        )
    status_code, response_dict = await make_metakg_query(
        APIInfo(request, response), tier=tier
    )
    return ORJSONResponse(response_dict, status_code=status_code)


@app.get(
    "/metadata/tier_{tier}",
    tags=["metadata"],
    response_description=OPENAPI_CONFIG.response_descriptions.metadata.get("200", ""),
)
async def metadata(
    request: Request, response: Response, tier: TierNumber
) -> ORJSONResponse:
    """Retrieve the metadata associated with a given Data Tier."""
    resolved = service_health.Snapshot().select_tier(None, tier, allow_fallback=False)
    if isinstance(resolved[0], HTTPStatus):
        status_code, detail = resolved
        raise HTTPException(status_code, detail=detail)
    status_code, response_dict = await make_metadata_query(
        APIInfo(request, response), tier=tier
    )
    return ORJSONResponse(response_dict, status_code=status_code)


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
    status_code, response_dict = await make_lookup_query(
        APIInfo(request, response), body=body
    )
    return ORJSONResponse(response_dict, status_code=status_code)
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
    status_code, response_dict = await make_lookup_query(
        APIInfo(request, response, background_tasks), body=body
    )
    return ORJSONResponse(response_dict, status_code=status_code)


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
async def asyncquery_status(request: Request, job_id: str) -> ORJSONResponse:
    """Get the status of an asynchronous query."""
    if status_code := service_health.Snapshot().http_status_for("/asyncquery_status"):
        raise HTTPException(
            status_code,
            detail=service_health.outage_detail(
                "MongoDB is unavailable; job state cannot be retrieved.",
                MongoClient(),
            ),
        )
    status_code, job_dict = await get_job_status(job_id.lower(), request)
    return ORJSONResponse(job_dict, status_code=status_code)


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
async def response(request: Request, job_id: str) -> ORJSONResponse:
    """Get the response for a query (or logs if it's in progress)."""
    if service_health.Snapshot().http_status_for("/response") is not None:
        raise HTTPException(
            HTTPStatus.FAILED_DEPENDENCY,
            detail=service_health.outage_detail(
                "MongoDB is unavailable; the response cannot be retrieved.",
                MongoClient(),
            ),
        )
    status_code, job_dict = await get_job_response(job_id.lower(), request)
    return ORJSONResponse(job_dict, status_code=status_code)


@app.post("/rehydrate")
async def rehydrate(body: dict[str, Any]) -> ORJSONResponse:
    """Passthrough rehydration to backend."""
    # TODO: use the appropriate tier based on parameters
    driver = tier_manager.get_driver(0)
    response_dict = await driver.run_query(body)
    for edge in (
        response_dict.get("message", {})
        .get("knowledge_graph", {})
        .get("edges", {})
        .values()
    ):
        append_aggregator_source(edge, "infores:retriever")
    return ORJSONResponse(response_dict)


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
            description="ID of a previously-run job to search for. Limits logs to those related to that job.",
            pattern=JOB_ID_PATTERN,
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
        raise HTTPException(
            HTTPStatus.NOT_FOUND, detail="Persisted logging not enabled."
        )
    if service_health.Snapshot().http_status_for("/logs") is not None:
        raise HTTPException(
            HTTPStatus.FAILED_DEPENDENCY,
            detail=service_health.outage_detail(
                "MongoDB is unavailable; logs cannot be served.",
                MongoClient(),
            ),
        )

    if job_id is not None:
        job_id = job_id.lower()

    if lookback is not None:
        start = datetime.now().astimezone() - timedelta(seconds=lookback * 60 * 60)

    elif not start and not job_id:  # Get all logs from last hour
        start = datetime.now().astimezone() - timedelta(hours=1)

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


@app.get(
    "/config",
    tags=["metadata"],
    response_description=OPENAPI_CONFIG.response_descriptions.config.get("200", ""),
)
async def config() -> ORJSONResponse:
    """Get the current config of the server."""
    config = yaml.safe_load(CONFIG.model_dump_json())
    sha, branch = get_version()
    if sha != "unknown":
        config["retriever_version"] = sha
        config["retriever_branch"] = branch
        config["retriever_version_link"] = (
            f"https://github.com/BioPack-team/retriever/tree/{sha}"
        )
    return ORJSONResponse(config)


# Set up Sentry and Otel
configure_telemetry(app)
