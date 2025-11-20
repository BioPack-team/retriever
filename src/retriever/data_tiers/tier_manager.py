import asyncio
import contextlib
from signal import SIGTERM

from loguru import logger

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.base_transpiler import Transpiler
from retriever.data_tiers.tier_0.base_query import Tier0Query
from retriever.data_tiers.tier_0.dgraph.driver import DgraphGrpcDriver
from retriever.data_tiers.tier_0.dgraph.query import DgraphQuery
from retriever.data_tiers.tier_0.dgraph.transpiler import DgraphTranspiler
from retriever.data_tiers.tier_1.elasticsearch.driver import ElasticSearchDriver
from retriever.data_tiers.tier_1.elasticsearch.transpiler import ElasticsearchTranspiler
from retriever.types.trapi_pydantic import TierNumber

BACKEND_DRIVERS = dict[str, DatabaseDriver](
    elasticsearch=ElasticSearchDriver(),
    dgraph=DgraphGrpcDriver(),
)

TRANSPILERS = dict[str, Transpiler](
    elasticsearch=ElasticsearchTranspiler(),
    dgraph=DgraphTranspiler(),
)

QUERY_HANDLERS = dict[str, type[Tier0Query]](
    dgraph=DgraphQuery,
)


def get_driver(tier: TierNumber) -> DatabaseDriver:
    """Get the configured driver for the given tier."""
    if tier == 0:
        return BACKEND_DRIVERS[CONFIG.tier0.backend]
    elif tier == 1:
        return BACKEND_DRIVERS[CONFIG.tier1.backend]
    else:
        raise NotImplementedError(f"Tier {tier} is not yet implemented.")


def get_transpiler(tier: TierNumber) -> Transpiler:
    """Get the configured transpiler for the given tier."""
    if tier == 0:
        return TRANSPILERS[CONFIG.tier0.backend]
    elif tier == 1:
        return TRANSPILERS[CONFIG.tier1.backend]
    else:
        raise NotImplementedError(f"Tier {tier} is not yet implemented.")


async def connect_drivers() -> None:
    """Connect all drivers simultaneously, logging failures."""
    connect_tasks: set[asyncio.Task[None]] = {
        asyncio.create_task(
            get_driver(i).connect(), name=f"Tier {i} backend connection"
        )
        for i in range(2)
    }

    def _cancel_tasks() -> None:
        for task in connect_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                task.cancel()

    # In case connect tasks run long and user wants to close
    asyncio.get_event_loop().add_signal_handler(SIGTERM, _cancel_tasks)

    successes = 0

    while connect_tasks:
        done, pending = await asyncio.wait(
            connect_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        for task in done:
            if task.exception():
                logger.exception(f"{task.get_name()} has failed.")
            else:
                successes += 1

        connect_tasks = pending

    if successes == 0:
        logger.warning(
            "All Data Tier backend connections have failed. Retriever won't be able to answer lookups."
        )


async def close_drivers() -> None:
    """Close all drivers simultaneously, skipping failures."""
    close_tasks: set[asyncio.Task[None]] = {
        asyncio.create_task(get_driver(i).close(), name=f"Tier {i}") for i in range(2)
    }

    while close_tasks:
        done, pending = await asyncio.wait(
            close_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        for task in done:
            if task.exception():
                logger.exception(f"Close operation for {task.get_name()} failed.")

        close_tasks = pending
