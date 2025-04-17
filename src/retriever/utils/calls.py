import sys

import httpx

from retriever.config.general import CONFIG

version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

BASIC_CLIENT = httpx.AsyncClient(
    timeout=30,
    follow_redirects=True,
    headers={"user-agent": f"Retriever/{CONFIG.instance_env} Python/{version}"},
)
