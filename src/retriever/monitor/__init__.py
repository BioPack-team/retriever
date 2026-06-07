"""Static-file dashboard served at /monitor.

Plain HTML / JS / CSS under `static/`, mounted via FastAPI's StaticFiles
from `server.py`. The browser polls the existing `/status/*` JSON API
directly - no server-side session state for the dashboard.
"""
