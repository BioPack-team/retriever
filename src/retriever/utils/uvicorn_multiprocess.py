import asyncio
import os
from collections.abc import Callable
from socket import socket
from typing import override

import click
from loguru import logger
from uvicorn.config import Config
from uvicorn.supervisors import Multiprocess


class AsyncMultiprocess(Multiprocess):
    """A subclass of Uvicorn's multiprocess supervisor that's async-friendly."""

    def __init__(
        self,
        config: Config,
        target: Callable[[list[socket] | None], None],
        sockets: list[socket],
    ) -> None:
        """Instantiate Multiprocess, replacing should_exit with an async event."""
        super().__init__(config, target, sockets)
        self.should_exit: asyncio.Event = asyncio.Event()

    @override
    async def run(self) -> None:  # pyright:ignore[reportIncompatibleMethodOverride] Intentionally made async
        message = f"Started parent process [{os.getpid()}]"
        color_message = "Started parent process [{}]".format(
            click.style(str(os.getpid()), fg="cyan", bold=True)
        )
        logger.info(message, extra={"color_message": color_message})

        self.init_processes()

        while True:
            try:
                async with asyncio.timeout(0.5):
                    if await self.should_exit.wait():
                        break
            except TimeoutError:
                self.handle_signals()
                self.keep_subprocess_alive()

        self.terminate_all()
        self.join_all()

        message = f"Stopping parent process [{os.getpid()}]"
        color_message = "Stopping parent process [{}]".format(
            click.style(str(os.getpid()), fg="cyan", bold=True)
        )
        logger.info(message, extra={"color_message": color_message})
