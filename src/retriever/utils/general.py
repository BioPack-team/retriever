import asyncio
import time
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
)
from pathlib import Path
from typing import Any, ClassVar, cast, override

from loguru import logger
from pydantic import BaseModel, Secret, SecretBytes, SecretStr
from pydantic_core import PydanticUndefinedType
from pydantic_settings import BaseSettings
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from retriever.types.general import JsonSerializable

yaml = YAML()


class Singleton(ABCMeta):
    """Singleton metaclass that ensures classes using it have only one instance."""

    _instances: ClassVar[dict["Singleton", "Singleton"]] = {}

    @override
    def __call__(cls, *args: Any, **kwargs: Any) -> "Singleton":
        """Ensure calls go to one instance."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)  # noqa:UP008
        return cls._instances[cls]


CommentedSerializable = JsonSerializable | list[CommentedMap] | dict[str, CommentedMap]


class CommentedSettings(BaseSettings):
    """Pydantic BaseSettings with support for yaml output with comment."""

    @staticmethod
    def recurse_common_types(obj: Any) -> CommentedSerializable | CommentedMap:
        """Recursively ensure an object is able to be dumped to yaml."""
        if isinstance(obj, BaseModel) or hasattr(obj, "model_fields"):
            return CommentedSettings.to_commented(obj)
        if isinstance(obj, str) or not isinstance(obj, Iterable | Mapping):
            if isinstance(obj, None | int | float | bool):
                return obj
            return str(obj)
        if not isinstance(obj, Mapping):
            return sorted([CommentedSettings.recurse_common_types(o) for o in obj])
        return {
            str(key): CommentedSettings.recurse_common_types(value)  # pyright:ignore[reportUnknownArgumentType]
            for key, value in obj.items()  # pyright:ignore[reportUnknownVariableType]
        }

    @staticmethod
    def to_commented(obj: BaseModel | type[BaseModel]) -> CommentedMap:
        """Recursively populate a commented mapping from a BaseModel."""
        commented = CommentedMap()
        is_basemodel = isinstance(obj, BaseModel)
        iterator = obj.__dict__.items() if is_basemodel else obj.model_fields.items()
        for field, value in iterator:
            if is_basemodel:
                adjusted_value = value
            else:
                adjusted_value = (
                    value.default_factory()  # pyright:ignore[reportCallIssue] No settings factory takes validated data
                    if value.default_factory
                    else value.default
                )
                if isinstance(adjusted_value, PydanticUndefinedType):
                    continue

            if isinstance(adjusted_value, BaseModel):
                adjusted_value = CommentedSettings.to_commented(adjusted_value)
            elif (
                isinstance(adjusted_value, list)
                and len(adjusted_value) > 0  # pyright:ignore[reportUnknownArgumentType]
                and isinstance(adjusted_value[0], BaseModel)
            ):
                # Cast because we can assume lists are of one type
                adjusted_value = cast(list[BaseModel], adjusted_value)
                adjusted_value = [
                    CommentedSettings.to_commented(v) for v in adjusted_value
                ]
            else:
                if isinstance(adjusted_value, Secret | SecretStr | SecretBytes):
                    adjusted_value = adjusted_value.get_secret_value()  # pyright:ignore[reportUnknownVariableType] Secrets use unknowns
                adjusted_value = CommentedSettings.recurse_common_types(adjusted_value)

            commented[field] = adjusted_value
            if (
                desc := (type(obj) if is_basemodel else obj)
                .model_fields[field]
                .description
            ):
                commented.yaml_add_eol_comment(comment=desc, key=field)  # pyright:ignore[reportUnknownMemberType]

        return commented

    @classmethod
    def write_default(cls, path: Path) -> None:
        """Write the settings defaults to a given path."""
        # commented = CommentedSettings.to_commented(cls)
        start_comment = "\n".join(
            [
                "Default configuration values.",
                "Managed by Retriever.",
                "Don't edit this file, it will be overwritten.",
                "Edit the appropriate config file instead.",
            ]
        )
        commented = CommentedSettings.to_commented(cls)

        commented.yaml_set_start_comment(start_comment)  # pyright:ignore[reportUnknownMemberType] ruamel uses unknowns
        yaml.dump(commented, path)  # pyright:ignore[reportUnknownMemberType] ruamel uses unknowns


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

    tasks: ClassVar[list[asyncio.Task[None]]] = []
    initialized: bool = False
    namespace: str = "AsyncDaemon"

    def initialize(self) -> None:
        """Start up long-running tasks."""
        if self.initialized:
            return
        logger.info(f"{self.__class__.__name__} starting tasks.")
        for i, func in enumerate(self.get_task_funcs()):
            self.tasks.append(
                asyncio.create_task(func(), name=f"{self.namespace}:task_{i}")
            )
        self.initialized = True

    def wrapup(self) -> None:
        """Cancel all long-running tasks."""
        logger.info(f"{self.__class__.__name__} wrapping up.")
        for task in self.tasks:
            task.cancel()

    @abstractmethod
    def get_task_funcs(self) -> list[Callable[[], Coroutine[None, None, None]]]:
        """Return a list of long-running task functions."""


class BatchedAction(AsyncDaemon):
    """A base class for queuing and batching a number of different sink types."""

    batch_size: int = 100
    queue_delay: float = 0.1
    flush_time: float = 1
    action_queues: ClassVar[dict[str, asyncio.Queue[Any]]] = {}

    @override
    def get_task_funcs(self) -> list[Callable[[], Coroutine[None, None, None]]]:
        return [self.process_queues]

    async def process_queues(self) -> None:
        """Periodically poll the queue for a payload and handle it accordingly."""
        last_flush: dict[str, float] = {}
        while True:
            try:
                for target, queue in self.action_queues.items():
                    now = time.time()

                    if queue.qsize() > 0 and (
                        target not in last_flush
                        or now - last_flush[target] >= self.flush_time
                    ):
                        batch = [queue.get_nowait() for _ in range(queue.qsize())]
                        last_flush[target] = now
                    elif queue.qsize() < self.batch_size:
                        continue
                    else:
                        batch = [
                            queue.get_nowait()
                            for _ in range(min(queue.qsize(), self.batch_size))
                        ]
                    await getattr(self, target)(batch)
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
