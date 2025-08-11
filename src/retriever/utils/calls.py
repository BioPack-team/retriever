import sys

import httpx

from retriever.config.general import CONFIG

version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


USER_AGENT = f"Retriever/{CONFIG.instance_env} Python/{version}"

callback_transport = httpx.AsyncHTTPTransport(retries=CONFIG.job.callback.retries)
CALLBACK_CLIENT = httpx.AsyncClient(
    timeout=CONFIG.job.callback.timeout,
    transport=callback_transport,
    follow_redirects=True,
    headers={"user-agent": f"Retriever/{CONFIG.instance_env} Python/{version}"},
)
