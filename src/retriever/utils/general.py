from __future__ import annotations

import asyncio
import time
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine
from typing import Any, ClassVar, override

from loguru import logger


class Singleton(ABCMeta):
    """Singleton metaclass that ensures classes using it have only one instance."""

    _instances: ClassVar[dict[Singleton, Singleton]] = {}

    @override
    def __call__(cls, *args: Any, **kwargs: Any) -> Singleton:
        """Ensure calls go to one instance."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)  # noqa:UP008
        return cls._instances[cls]


async def await_next[T](iterator: AsyncIterator[T]) -> T:
    """Create a coroutine awaiting next value in iterator."""
    return await iterator.__anext__()


def as_task[T](iterator: AsyncIterator[T]) -> asyncio.Task[T]:
    """Create a task which resolves to the iterator's next result."""
    return asyncio.create_task(await_next(iterator))


class EmptyIteratorError(Exception):
    """An iterator terminated before yielding anything."""

    def __init__(self, message: str, index: int) -> None:
        """Instantiate an EmptyIteratorError."""
        super().__init__(message)
        self.iterator: int = index


async def merge_iterators[T](
    *iterators: AsyncIterator[T], raise_on_empty: bool = False
) -> AsyncIterable[T]:
    """Merge multiple async iterators, yielding values as they are completed.

    Based on Alex Peter's solution: https://stackoverflow.com/a/76643550
    """
    next_tasks = {iterator: as_task(iterator) for iterator in iterators}
    backmap = {v: k for k, v in next_tasks.items()}
    yielded = dict.fromkeys(iterators, False)

    while next_tasks:
        done, _ = await asyncio.wait(
            next_tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            iterator = backmap[task]

            try:
                yield task.result()
                yielded[iterator] = True
            except StopAsyncIteration as e:
                del next_tasks[iterator]
                del backmap[task]
                if raise_on_empty and not yielded[iterator]:
                    raise EmptyIteratorError(
                        f"Iterator {iterators.index(iterator)} terminated before yielding anything.",
                        iterators.index(iterator),
                    ) from e
            except Exception as e:
                raise e
            else:
                next_task = as_task(iterator)
                next_tasks[iterator] = next_task
                backmap[next_task] = iterator


class AsyncDaemon(ABC, metaclass=Singleton):
    """A base class for handling a number of lifetime-running async tasks."""

    tasks: list[asyncio.Task[None]]
    initialized: bool

    def __init__(self) -> None:
        """Instantiate a new AsyncDaemon."""
        super().__init__()
        self.tasks = []
        self.initialized = False

    async def initialize(self) -> None:
        """Start up long-running tasks."""
        if self.initialized:
            return
        logger.info(f"{self.__class__.__name__} starting tasks.")
        for i, func in enumerate(self.get_task_funcs()):
            self.tasks.append(
                asyncio.create_task(func(), name=f"{self.__class__.__name__}:task_{i}")
            )
        self.initialized = True

    async def wrapup(self) -> None:
        """Cancel all long-running tasks."""
        logger.info(f"{self.__class__.__name__} wrapping up.")
        for task in self.tasks:
            task.cancel()

    @abstractmethod
    def get_task_funcs(self) -> list[Callable[[], Coroutine[None, None, None]]]:
        """Return a list of long-running task functions."""


class BatchedAction(AsyncDaemon, metaclass=Singleton):
    """A base class for queuing and batching a number of different sink types."""

    batch_size: int = 100  # How many to batch together
    queue_delay: float = 0.1  # How long between queue checks
    flush_time: float = (
        1  # How long until the queue should be flushed, even below batch size
    )
    multibatch: bool = False  # If true, drain the queue completely every interval and run all batches simultaneously
    action_queues: dict[str, asyncio.Queue[Any]]

    def __init__(self) -> None:
        """Instantiate a new BatchedAction."""
        self.action_queues = {}
        super().__init__()

    @override
    def get_task_funcs(self) -> list[Callable[[], Coroutine[None, None, None]]]:
        return [self.process_queues]

    async def process_queues(self) -> None:
        """Periodically poll the queue for a payload and handle it accordingly."""
        last_flush: dict[str, float] = {}
        while True:
            try:
                for task in self.tasks:  # Clean up finished batch tasks
                    if task.cancelled() or task.done():
                        self.tasks.remove(task)
                for target, queue in self.action_queues.items():
                    now = time.time()

                    should_flush = not (
                        target in last_flush
                        and now - last_flush[target] < self.flush_time
                    )
                    # Use a static batch size so as to not keep looping as new items
                    # are queued
                    queue_size = queue.qsize()
                    if queue_size == 0 or (
                        queue_size < self.batch_size and not should_flush
                    ):
                        continue

                    batches = list[list[Any]]()

                    while queue_size > 0:
                        batch_size = min(queue_size, self.batch_size)
                        batches.append([queue.get_nowait() for _ in range(batch_size)])
                        queue_size -= batch_size
                        if not self.multibatch:
                            break

                    if should_flush:
                        last_flush[target] = now

                    for batch in batches:
                        self.tasks.append(
                            asyncio.create_task(getattr(self, target)(batch))
                        )
                await asyncio.sleep(self.queue_delay)

            except asyncio.QueueEmpty:
                try:
                    await asyncio.sleep(self.queue_delay)
                    continue
                except asyncio.CancelledError:
                    break

            except asyncio.CancelledError:
                break

    def put(self, target: str, payload: Any) -> None:
        """Put a payload in the target action queue."""
        if not hasattr(self, target):
            raise NotImplementedError(f"Target action {target} is not implemented!")

        if target not in self.action_queues:
            self.action_queues[target] = asyncio.Queue()

        try:
            self.action_queues[target].put_nowait(payload)
        except (asyncio.QueueShutDown, asyncio.QueueFull):
            return
