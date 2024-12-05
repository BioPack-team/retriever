from datetime import datetime
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

from retriever.config.cors import CorsSettings

def add_cors_manually(APP, request, response, cors_options):
    """
    Add CORS to a response manually
    Based on https://github.com/tiangolo/fastapi/issues/775
    """

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
        cors = CORSMiddleware(APP, **cors_options)

        # Logic directly from Starlette's CORSMiddleware:
        # https://github.com/encode/starlette/blob/master/starlette/middleware/cors.py#L152

        response.headers.update(cors.simple_headers)
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

def add_middleware(APP):

    async def catch_exceptions(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as error:
            response = JSONResponse(
                {
                    "message": {},
                    "logs": [
                        {
                            "message": f"Unhandled exepction in Retriever: {repr(error)}",
                            "level": "ERROR",
                            "timestamp": datetime.now().isoformat(),
                            "stack": traceback.format_exc(),
                        }
                    ],
                },
                status_code=500,
            )
            add_cors_manually(APP, request, response, CorsSettings().dict())
            return response

    APP.middleware("http")(catch_exceptions)
