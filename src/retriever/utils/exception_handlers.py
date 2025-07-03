import asyncio
import functools
import traceback
from datetime import datetime
from typing import Any, Callable, cast

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from retriever.config.general import CONFIG


async def ensure_cors(app: FastAPI, request: Request, exc: Exception) -> Response:
    """Exception handler that ensures CORS information is kept in response.

    Based on https://github.com/tiangolo/fastapi/issues/775
    """
    response = JSONResponse(
        {
            "message": {},
            "logs": [
                {
                    "message": f"Unhandled exepction in Retriever: {exc!r}",
                    "level": "ERROR",
                    "timestamp": datetime.now()
                    .astimezone()
                    .isoformat(timespec="milliseconds"),
                    "stack": traceback.format_exc(),
                },
            ],
        },
        status_code=500,
    )

    # Since the CORSMiddleware is not executed when an unhandled server exception
    # occurs, we need to manually set the CORS headers ourselves if we want the FE
    # to receive a proper JSON 500, opposed to a CORS error.
    # Setting CORS headers on server errors is a bit of a philosophical topic of
    # discussion in many frameworks, and it is currently not handled in FastAPI.
    # See dotnet core for a recent discussion, where ultimately it was
    # decided to return CORS headers on server failures:
    # https://github.com/dotnet/aspnetcore/issues/2378
    origin = request.headers.get("origin")

    if origin:
        # Have the middleware do the heavy lifting for us to parse
        # all the config, then update our response headers
        cors = CORSMiddleware(
            app,
            allow_origins=CONFIG.cors.allow_origins,
            allow_credentials=CONFIG.cors.allow_credentials,
            allow_methods=CONFIG.cors.allow_methods,
            allow_headers=CONFIG.cors.allow_headers,
        )

        # Logic directly from Starlette's CORSMiddleware:
        # https://github.com/encode/starlette/blob/master/starlette/middleware/cors.py#L152

        response.headers.update(cast(dict[str, str], cors.simple_headers))
        has_cookie = "cookie" in request.headers

        # If request includes any cookie headers, then we must respond
        # with the specific origin instead of '*'.
        if cors.allow_all_origins and has_cookie:
            response.headers["Access-Control-Allow-Origin"] = origin

        # If we only allow specific origins, then we have to mirror back
        # the Origin header in the response.
        elif not cors.allow_all_origins and cors.is_allowed_origin(origin=origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers.add_vary_header("Origin")

    return response
