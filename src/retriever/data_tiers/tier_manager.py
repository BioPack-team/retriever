import asyncio
import contextlib
from signal import SIGTERM

from loguru import logger

from retriever.config.general import CONFIG
from retriever.data_tiers.base_driver import DatabaseDriver
from retriever.data_tiers.base_transpiler import Transpiler
from retriever.data_tiers.tier_0.base_query import Tier0Query
from retriever.data_tiers.tier_0.gandalf.driver import GandalfDriver
from retriever.data_tiers.tier_0.gandalf.query import GandalfQuery
from retriever.data_tiers.tier_1.elasticsearch.driver import ElasticSearchDriver
from retriever.data_tiers.tier_1.elasticsearch.transpiler import ElasticsearchTranspiler
from retriever.types.trapi_pydantic import TierNumber

BACKEND_DRIVERS = dict[str, type[DatabaseDriver]](
    elasticsearch=ElasticSearchDriver,
    gandalf=GandalfDriver,
)

TRANSPILERS = dict[str, type[Transpiler]](
    elasticsearch=ElasticsearchTranspiler,
)

QUERY_HANDLERS = dict[str, type[Tier0Query]](
    gandalf=GandalfQuery,
)

IMPLEMENTED_TIERS: frozenset[TierNumber] = frozenset((0, 1))
"""Tiers `get_driver` / `get_transpiler` resolve without raising."""


def get_driver(tier: TierNumber) -> DatabaseDriver:
    """Get the configured driver for the given tier."""
    if tier == 0:
        return BACKEND_DRIVERS[CONFIG.tier0.backend]()
    elif tier == 1:
        return BACKEND_DRIVERS[CONFIG.tier1.backend]()
    else:
        raise NotImplementedError(f"Tier {tier} is not yet implemented.")


def get_transpiler(tier: TierNumber) -> Transpiler:
    """Get the configured transpiler for the given tier."""
    if tier == 0:
        return TRANSPILERS[CONFIG.tier0.backend]()
    elif tier == 1:
        return TRANSPILERS[CONFIG.tier1.backend]()
    else:
        raise NotImplementedError(f"Tier {tier} is not yet implemented.")


async def initialize_drivers() -> None:
    """Initialize all drivers simultaneously, logging failures."""
    init_tasks: set[asyncio.Task[None]] = {
        asyncio.create_task(
            get_driver(i).initialize(), name=f"Tier {i} backend initialize"
        )
        for i in range(2)
    }

    def _cancel_tasks() -> None:
        for task in init_tasks:
            with contextlib.suppress(asyncio.CancelledError):
                task.cancel()

    # In case init tasks run long and user wants to close
    asyncio.get_event_loop().add_signal_handler(SIGTERM, _cancel_tasks)

    successes = 0

    while init_tasks:
        done, pending = await asyncio.wait(
            init_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        for task in done:
            if task.exception():
                logger.exception(f"{task.get_name()} has failed.")
            else:
                successes += 1

        init_tasks = pending

    if successes == 0:
        logger.warning(
            "All Data Tier backend connections have failed. Retriever won't be able to answer lookups."
        )


def enable_periodic_healthchecks(interval: float = 60.0) -> None:
    """Switch each tier driver to periodic-ping mode; call before `initialize_drivers`."""
    for tier in range(2):
        get_driver(tier).healthcheck_interval = interval


async def wrapup_drivers() -> None:
    """Wrap up all drivers simultaneously, skipping failures."""
    close_tasks: set[asyncio.Task[None]] = {
        asyncio.create_task(get_driver(i).wrapup(), name=f"Tier {i}") for i in range(2)
    }

    while close_tasks:
        done, pending = await asyncio.wait(
            close_tasks, return_when=asyncio.FIRST_EXCEPTION
        )

        for task in done:
            if task.exception():
                logger.exception(f"Close operation for {task.get_name()} failed.")

        close_tasks = pending
