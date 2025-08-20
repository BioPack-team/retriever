import asyncio
from abc import ABCMeta
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path
from typing import Any, ClassVar, cast, override

from pydantic import BaseModel, Secret, SecretBytes, SecretStr
from pydantic_core import PydanticUndefinedType
from pydantic_settings import BaseSettings
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

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


class CommentedSettings(BaseSettings):
    """Pydantic BaseSettings with support for yaml output with comment."""

    @staticmethod
    def to_commented(obj: BaseModel | type[BaseModel]) -> CommentedMap:
        """Recursively populate a commented mapping from a BaseModel."""
        commented = CommentedMap()
        is_instance = isinstance(obj, BaseModel)
        iterator = obj.__dict__.items() if is_instance else obj.model_fields.items()
        for field, value in iterator:
            if is_instance:
                adjusted_value = value
            else:
                adjusted_value = (
                    value.default_factory()  # pyright:ignore[reportCallIssue] No settings factory takes validated data
                    if value.default_factory
                    else value.default
                )
                if isinstance(adjusted_value, PydanticUndefinedType):
                    continue

            if isinstance(adjusted_value, Secret | SecretStr | SecretBytes):
                adjusted_value = adjusted_value.get_secret_value()  # pyright:ignore[reportUnknownVariableType] Secrets use unknowns
            elif isinstance(adjusted_value, BaseModel):
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

            commented[field] = adjusted_value
            if (
                desc := (type(obj) if is_instance else obj)
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


async def merge_iterators[T](*iterators: AsyncIterator[T]) -> AsyncIterable[T]:
    """Merge multiple async iterators, yielding values as they are completed.

    Based on Alex Peter's solution: https://stackoverflow.com/a/76643550
    """
    next_tasks = {iterator: as_task(iterator) for iterator in iterators}
    backmap = {v: k for k, v in next_tasks.items()}

    while next_tasks:
        done, _ = await asyncio.wait(
            next_tasks.values(), return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            iterator = backmap[task]

            try:
                yield task.result()
            except StopAsyncIteration:
                del next_tasks[iterator]
                del backmap[task]
            except Exception as e:
                raise e
            else:
                next_task = as_task(iterator)
                next_tasks[iterator] = next_task
                backmap[next_task] = iterator
