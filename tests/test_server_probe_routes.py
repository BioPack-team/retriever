"""Tests for /health and /config.

Both are exercised over ASGITransport *without* running the lifespan, so no
Mongo, Redis or tier backend is required - which is also the property that
makes /health a valid per-pod probe in the first place.
"""

from __future__ import annotations

import orjson
import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel, SecretStr

from retriever.config.general import CONFIG
from retriever.server import app


@pytest.fixture
def client() -> AsyncClient:
    """An AsyncClient bound to the app, with the lifespan deliberately unrun."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_health_needs_no_backend(client: AsyncClient) -> None:
    """/health answers 200 with no dependency initialized."""
    async with client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_stays_out_of_the_public_schema() -> None:
    """/health is infrastructure, not part of the documented API."""
    paths = app.openapi()["paths"]
    assert "/health" not in paths
    assert "/config" in paths  # ...unlike /config, which is documented.


@pytest.mark.asyncio
async def test_config_reports_version(client: AsyncClient) -> None:
    """/config serves the config plus git metadata when the SHA is known."""
    async with client:
        response = await client.get("/config")

    assert response.status_code == 200
    body = response.json()
    assert body["debug"] == CONFIG.debug
    if "retriever_version" in body:
        assert body["retriever_version_link"].endswith(body["retriever_version"])


def test_config_still_redacts_secrets() -> None:
    """The /config contract is "with secrets removed" - keep it that way.

    `mode="json"` is load-bearing here: a plain `model_dump()` leaves live
    SecretStr objects in the dict with their plaintext retrievable.
    """
    secrets = _secret_fields(CONFIG)
    if not secrets:
        pytest.skip("no populated SecretStr in this environment's config")

    dumped = orjson.dumps(CONFIG.model_dump(mode="json")).decode()
    for secret in secrets:
        assert secret.get_secret_value() not in dumped


def _secret_fields(model: object) -> list[SecretStr]:
    """Collect every populated SecretStr reachable from a pydantic model."""
    if not isinstance(model, BaseModel):
        return []
    found: list[SecretStr] = []
    for value in dict(model).values():
        if isinstance(value, SecretStr) and value.get_secret_value():
            found.append(value)
        else:
            found.extend(_secret_fields(value))
    return found
