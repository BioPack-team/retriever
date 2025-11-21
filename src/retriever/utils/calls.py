import sys

import httpx

from retriever.config.general import CONFIG

version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


USER_AGENT = f"Retriever/{CONFIG.instance_env} Python/{version}"


def get_callback_client() -> httpx.AsyncClient:
    """Get a client with appropriate settings for making callbacks."""
    transport = httpx.AsyncHTTPTransport(retries=CONFIG.job.callback.retries)
    return httpx.AsyncClient(
        timeout=CONFIG.job.callback.timeout,
        transport=transport,
        follow_redirects=True,
        headers={"user-agent": f"Retriever/{CONFIG.instance_env} Python/{version}"},
    )


def get_metadata_client() -> httpx.AsyncClient:
    """Get a client with appropriate settings for obtaining outside metadata."""
    transport = httpx.AsyncHTTPTransport(retries=CONFIG.job.callback.retries)
    return httpx.AsyncClient(
        timeout=CONFIG.job.metakg.acquire_timeout,
        transport=transport,
        follow_redirects=True,
        headers={"user-agent": f"Retriever/{CONFIG.instance_env} Python/{version}"},
    )
