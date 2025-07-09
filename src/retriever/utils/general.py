from abc import ABCMeta
from typing import Any, ClassVar, override


class Singleton(ABCMeta):
    """Singleton metaclass that ensures classes using it have only one instance."""

    _instances: ClassVar[dict["Singleton", "Singleton"]] = {}

    @override
    def __call__(cls, *args: Any, **kwargs: Any) -> "Singleton":
        """Ensure calls go to one instance."""
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)  # noqa:UP008
        return cls._instances[cls]
