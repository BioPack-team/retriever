import asyncio
from typing import Any

from loguru import logger as log
from opentelemetry import trace
from saq.types import Context

from retriever.type_defs import Query
from retriever.utils.logging import add_job_trapi_sink
from retriever.utils.telemetry import align_context

tracer = trace.get_tracer("lookup.execution.tracer")


async def lookup(ctx: Context, query: Query) -> dict[str, Any]:
    """Execute a lookup query."""
    job = ctx.get("job")
    if not job:
        return {"status": "Error (no job id)"}
    align_context(job)  # Ensure Otel context is carried appropriately

    # TODO implement actual query handling
    with tracer.start_as_current_span("execute_lookup_query"):
        job_log, logs, remove_sink = add_job_trapi_sink(job)
        job_log.info(f"got request with arg {job.key}...")
        with tracer.start_as_current_span("long_intermediate_step"):
            await asyncio.sleep(3)
            job_log.info("Some intermediate step")
        with tracer.start_as_current_span("finish_step"):
            await asyncio.sleep(3)
        job_log.info("finished waiting.")
        await log.complete()
        remove_sink()
        # raise ValueError("Some test error")
        return {
            "job_id": job.key,
            "status": "Complete",
            "logs": list(logs),
            "message": "this'll be a TRAPI message soon!",
        }
