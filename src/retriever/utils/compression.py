"""zstd request decompression middleware and content-negotiation helpers."""

import io
from http import HTTPStatus

import zstandard
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

ZSTD_DECOMPRESSOR = zstandard.ZstdDecompressor()

ZSTD_ENCODINGS = frozenset({"zstd", "zst"})
"""`Content-Encoding` / `Accept-Encoding` codings treated as zstd."""


def accepts_zstd(accept_encoding: str) -> bool:
    """Return whether an `Accept-Encoding` header value permits zstd.

    Parses the comma-separated codings and their optional `q` weights, treating
    zstd as acceptable only when listed with a non-zero weight; `zstd;q=0` is an
    explicit refusal.
    """
    for part in accept_encoding.lower().split(","):
        coding, _, params = part.strip().partition(";")
        if coding.strip() not in ZSTD_ENCODINGS:
            continue
        weight = 1.0
        for param in params.split(";"):
            key, _, value = param.partition("=")
            if key.strip() == "q":
                try:
                    weight = float(value.strip())
                except ValueError:
                    weight = 0.0
        if weight > 0:
            return True
    return False


class ZstdRequestMiddleware:
    """Decompress `Content-Encoding: zstd` request bodies and cap body size.

    zstd bodies are decompressed transparently and rejected with 413 when they
    expand past the configured maximum, guarding against decompression bombs.
    Bodies in any other encoding pass through, but are rejected with 413 when
    their declared `Content-Length` exceeds the same maximum.
    """

    def __init__(self, app: ASGIApp, max_request_size: int) -> None:
        """Wrap the downstream ASGI app.

        Args:
            app: The downstream ASGI application.
            max_request_size: Maximum request body size in bytes, measured after
                decompression for zstd bodies. Larger bodies are rejected with 413.
        """
        self.app: ASGIApp = app
        self.max_request_size: int = max_request_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce the size cap and decompress before handing off to the app."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        encoding = headers.get("content-encoding", "").lower()
        if encoding not in ZSTD_ENCODINGS:
            # Not zstd: cap the raw body by its declared length. (A zstd body's
            # Content-Length is the compressed size, so it is checked against the
            # decompressed output below instead.)
            if _declared_length(headers) > self.max_request_size:
                await self._reject_too_large(scope, receive, send)
                return
            await self.app(scope, receive, send)
            return

        compressed = bytearray()
        more_body = True
        while more_body:
            message = await receive()
            compressed.extend(message.get("body", b""))
            more_body = message.get("more_body", False)

        try:
            # Read one byte past the cap so an oversized body is detected without
            # decompressing the whole (potentially bomb-sized) payload.
            reader = ZSTD_DECOMPRESSOR.stream_reader(io.BytesIO(bytes(compressed)))
            body = reader.read(self.max_request_size + 1)
        except zstandard.ZstdError:
            await Response(
                "Malformed zstd-compressed request body.",
                status_code=HTTPStatus.BAD_REQUEST,
            )(scope, receive, send)
            return

        if len(body) > self.max_request_size:
            await self._reject_too_large(scope, receive, send)
            return

        # The body is now plain bytes: drop the stale encoding header and correct
        # the length so downstream parsing reads the full decompressed payload.
        mutable_headers = MutableHeaders(scope=scope)
        del mutable_headers["content-encoding"]
        mutable_headers["content-length"] = str(len(body))

        await self.app(scope, _replay(body), send)

    async def _reject_too_large(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Send a 413 for a request body that exceeds the configured maximum."""
        await Response(
            "Request body exceeds the maximum allowed size.",
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )(scope, receive, send)


def _declared_length(headers: Headers) -> int:
    """Parse the `Content-Length` header, treating absent/invalid values as zero."""
    raw = headers.get("content-length")
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


def _replay(body: bytes) -> Receive:
    """Build a `receive` channel that yields `body` as one request event."""
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive
