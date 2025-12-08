import asyncio
import os
import signal
import sys
from multiprocessing import Process

import uvloop
from loguru import logger
from uvicorn.supervisors.multiprocess import SIGNALS

from retriever.config.logger import configure_logging
from retriever.data_tiers import tier_manager
from retriever.metakg.metakg import MetaKGManager
from retriever.utils.logs import add_mongo_sink
from retriever.utils.mongo import MONGO_CLIENT, MONGO_QUEUE
from retriever.utils.redis import REDIS_CLIENT
from retriever.utils.telemetry import configure_telemetry


@logger.catch(
    message="Unhandled exception in background process.", onerror=lambda e: sys.exit(1)
)
async def _background_async() -> None:
    # /// SETUP ///

    await MONGO_CLIENT.initialize()
    await MONGO_QUEUE.start_process_task()
    add_mongo_sink()
    await REDIS_CLIENT.initialize()
    await tier_manager.connect_drivers()
    metakg_manager = MetaKGManager(leader=True)
    await metakg_manager.initialize()

    # /// MAIN LOOP ///

    logger.info("Background process loop started.")
    while True:
        try:
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            break

    # /// WRAPUP ///

    await metakg_manager.wrapup()
    await REDIS_CLIENT.close()
    await MONGO_QUEUE.stop_process_task()
    await MONGO_CLIENT.close()


def background_process() -> None:
    """A simple sync wrapper for the background process."""
    os.environ["PYTHONHASHSEED"] = "0"  # So reasoner_pydantic hashing is deterministic
    configure_logging()
    configure_telemetry()
    uvloop.run(_background_async())


class BackgroundProcessManager:
    """A helper class that maintains a background process.

    Based on Uvicorn's Multiprocess supervisor.
    """

    def __init__(self) -> None:
        """Instantiate an instance."""
        self.process: Process
        self.should_exit: asyncio.Event
        self.process_monitor_task: asyncio.Task[None]
        self.signal_queue: list[int] = []
        for sig in SIGNALS:
            signal.signal(sig, lambda sig, frame: self.signal_queue.append(sig))

    async def monitor_process(self) -> None:
        """Monitor the process and restart if it failed."""
        logger.info("Monitoring background process...")
        while True:
            try:
                async with asyncio.timeout(0.5):
                    if await self.should_exit.wait():
                        break
            except TimeoutError:
                self.handle_signals()
                await self.keep_subprocess_alive()
            except asyncio.CancelledError:
                break

    async def keep_subprocess_alive(self) -> None:
        """Ensure process is restarted if it has died."""
        if self.should_exit.is_set():
            return  # parent process is exiting, no need to keep subprocess alive

        if self.process.is_alive():
            return

        self.process.kill()  # process is hung, kill it
        self.process.join()

        if self.should_exit.is_set():
            return

        logger.info(f"Background process [{self.process.pid}] died")
        await self.start_process()

    async def start_process(self) -> None:
        """Start a new background process and create a task to monitor it."""
        logger.debug("Starting new background process...")
        self.should_exit = asyncio.Event()
        self.process = Process(target=background_process)
        self.process.start()
        self.process_monitor_task = asyncio.create_task(self.monitor_process())

    async def wrapup(self) -> None:
        """Stop monitoring the process and signal for it to close."""
        self.process_monitor_task.cancel()
        self.should_exit.set()
        self.process.join()
        logger.info("Background process stopped.")

    def handle_signals(self) -> None:
        """Handle any received signals."""
        for sig in tuple(self.signal_queue):
            self.signal_queue.remove(sig)
            sig_name = SIGNALS[sig]
            sig_handler = getattr(self, f"handle_{sig_name.lower()}", None)
            if sig_handler is not None:
                sig_handler()
            else:
                logger.debug(
                    f"Received signal {sig_name}, but no handler is defined for it."
                )

    def handle_int(self) -> None:
        """Terminate."""
        self.should_exit.set()

    def handle_term(self) -> None:
        """Terminate."""
        self.should_exit.set()

    def handle_break(self) -> None:
        """Terminate."""
        self.should_exit.set()
