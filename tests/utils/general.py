from typing import Any

from retriever.types.general import LogLevel
from retriever.utils.logs import TRAPILogger, format_trapi_log


def mock_inner_log(
    self: TRAPILogger, level: LogLevel, message: str, **_kwargs: Any
) -> None:
    """A version of the inner TRAPILogger._log() method which doesn't log to loguru."""

    # Implicitly drop TRACE logs from TRAPI logs.
    # These should only be used in extensive debugging on a local instance.
    if level.lower() != "trace":
        self.log_deque.append(format_trapi_log(level, message))
