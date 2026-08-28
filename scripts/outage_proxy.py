"""Toggleable HTTP reverse proxy for simulating mid-run backend outages.

Sits between Retriever and a tier backend - both Gandalf (tier 0) and
Elasticsearch (tier 1) speak HTTP, so the same proxy fronts either. In `pass`
mode it transparently forwards every request upstream; a control call flips it
to fail (and another flips it back), so you can make a backend go down and
recover while the server is running - to exercise whatever depends on a tier
being up or down: fallback routing, `/status`, health propagation, timeouts.

Point a tier at the proxy instead of the real backend (see tier0_proxy.py /
tier1_proxy.py for the exact overrides), keep the proxy's `--upstream` aimed at
the real backend, and drive it with the control routes below.

Control routes (under a reserved prefix that never collides with backend paths):

    POST /__outage__/down?mode=error|hang   start failing
    POST /__outage__/up                      resume forwarding
    GET  /__outage__/state                   {"mode": ..., "upstream": ...}

Modes:
    error  return 503 to every proxied request (backend up but erroring)
    hang   sleep past the driver's connect timeout, then 504 (unreachable)

For a hard connection-refused outage instead, just Ctrl-C the proxy.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal, cast

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

CONTROL_PREFIX = "/__outage__"

Mode = Literal["pass", "error", "hang"]
OutageMode = Literal["error", "hang"]

# Connection-level headers that must not be forwarded verbatim.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
    }
)
# httpx already decodes the upstream body and sets a fresh length, so echoing
# the upstream's own values would corrupt the response.
_RESP_STRIP = _HOP_BY_HOP | {"content-encoding", "content-length"}
_REQ_STRIP = _HOP_BY_HOP | {"content-length"}

_PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


def build_app(*, upstream: str, name: str, hang_seconds: float) -> FastAPI:
    """Build the reverse-proxy app forwarding to `upstream`, gated by a mode flag."""
    upstream = upstream.rstrip("/")
    state: dict[str, Mode] = {"mode": "pass"}
    client: httpx.AsyncClient | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal client
        client = httpx.AsyncClient(timeout=None, follow_redirects=False)
        print(f"[{name}] forwarding to {upstream}")
        yield
        await client.aclose()

    app = FastAPI(lifespan=lifespan, title=f"{name} outage proxy")

    @app.post(f"{CONTROL_PREFIX}/down")
    async def down(mode: OutageMode = "error") -> Response:
        state["mode"] = mode
        print(f"[{name}] >>> OUTAGE (mode={mode})")
        return JSONResponse({"mode": mode})

    @app.post(f"{CONTROL_PREFIX}/up")
    async def up() -> Response:
        state["mode"] = "pass"
        print(f"[{name}] <<< RECOVERED")
        return JSONResponse({"mode": "pass"})

    @app.get(f"{CONTROL_PREFIX}/state")
    async def get_state() -> Response:
        return JSONResponse({"mode": state["mode"], "upstream": upstream})

    @app.api_route("/{path:path}", methods=_PROXY_METHODS)
    async def proxy(request: Request, path: str) -> Response:
        mode = state["mode"]
        if mode == "error":
            return Response(b"outage-proxy: simulated outage", status_code=503)
        if mode == "hang":
            await asyncio.sleep(hang_seconds)
            return Response(b"outage-proxy: simulated hang", status_code=504)

        assert client is not None
        body = await request.body()
        req = client.build_request(
            request.method,
            f"{upstream}/{path}",
            params=request.url.query,
            headers=[
                (k, v)
                for k, v in request.headers.items()
                if k.lower() not in _REQ_STRIP
            ],
            content=body,
        )
        try:
            upstream_resp = await client.send(req)
        except httpx.HTTPError as exc:
            return Response(
                f"outage-proxy: upstream error: {exc}".encode(), status_code=502
            )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers={
                k: v
                for k, v in upstream_resp.headers.items()
                if k.lower() not in _RESP_STRIP
            },
        )

    return app


def run(*, name: str, upstream: str, host: str, port: int, hang_seconds: float) -> None:
    """Print usage hints and serve the proxy until interrupted."""
    app = build_app(upstream=upstream, name=name, hang_seconds=hang_seconds)
    base = f"http://{host}:{port}"
    print(f"[{name}] outage proxy listening on {base}  ->  {upstream}")
    print(f"[{name}]   knock it down:  curl -X POST '{base}{CONTROL_PREFIX}/down'")
    print(
        f"[{name}]   (or hang mode): curl -X POST '{base}{CONTROL_PREFIX}/down?mode=hang'"
    )
    print(f"[{name}]   bring it back:  curl -X POST '{base}{CONTROL_PREFIX}/up'")
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    """Generic entrypoint; the tier scripts wrap this with per-tier defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        required=True,
        help="Base URL of the REAL backend to forward to, e.g. http://localhost:9200",
    )
    parser.add_argument("--name", default="proxy", help="Label used in log output.")
    parser.add_argument("--host", default="127.0.0.1", help="Listen host.")
    parser.add_argument("--port", type=int, required=True, help="Listen port.")
    parser.add_argument(
        "--hang-seconds",
        type=float,
        default=8.0,
        help="Seconds to stall in `hang` mode (keep above the driver's connect timeout).",
    )
    args = parser.parse_args()
    run(
        name=cast(str, args.name),
        upstream=cast(str, args.upstream),
        host=cast(str, args.host),
        port=cast(int, args.port),
        hang_seconds=cast(float, args.hang_seconds),
    )


if __name__ == "__main__":
    main()
