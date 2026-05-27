"""Fixtures + seeding helpers for live MongoClient tests.

Lives under tests/utils/ which is in `norecursedirs` so pytest won't try
to collect this as test code. Imported from live-marked test modules.

Strategy: keep the MongoClient singleton but redirect its job-collection
accessor at a uniquely-named test DB for the session, seed it with a
small varied set of docs, and drop the DB on teardown.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

import pytest_asyncio
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import InsertOne

from retriever.utils.mongo import CODEC_OPTIONS, MongoClient

TEST_DB_PREFIX = "retriever_persist_test_"


@pytest_asyncio.fixture
async def test_mongo(monkeypatch):  # noqa: ANN001
    """Yield a MongoClient pointed at a seeded, throwaway test database."""
    client = MongoClient()
    await client.initialize()

    test_db_name = f"{TEST_DB_PREFIX}{uuid.uuid4().hex[:8]}"
    test_db = client.client[test_db_name]

    def patched_get_job_collection() -> (
        tuple[
            AsyncIOMotorCollection[dict[str, Any]],
            AsyncIOMotorCollection[dict[str, Any]],
        ]
    ):
        return (
            test_db.get_collection("job_status", codec_options=CODEC_OPTIONS),
            test_db.get_collection("job_docs", codec_options=CODEC_OPTIONS),
        )

    monkeypatch.setattr(client, "get_job_collection", patched_get_job_collection)

    status_collection, _ = patched_get_job_collection()
    await status_collection.create_index("job_id", unique=True, background=True)
    await status_collection.create_index("created", background=True)
    await status_collection.create_index("completed", background=True)

    await _seed(status_collection)

    try:
        yield client
    finally:
        await client.client.drop_database(test_db_name)


async def _seed(collection: AsyncIOMotorCollection[dict[str, Any]]) -> None:
    """Seed a varied set of job_status docs.

    Includes terminal-success / failure / non-terminal jobs across two
    submitters and two tiers, with varied creation times and durations.
    """
    now = datetime.now().astimezone()
    docs: list[dict[str, Any]] = []

    # Five completed jobs from alice, tier 0, durations 1s..5s, created in
    # the last hour.
    for i in range(5):
        created = now - timedelta(minutes=10 * (i + 1))
        duration = timedelta(seconds=i + 1)
        docs.append(
            _status_doc(
                status="Complete",
                submitter="alice",
                tiers=[0],
                created=created,
                completed=created + duration,
                results=10 + i,
            )
        )

    # Three failed jobs from bob, tier 1, mixed failure modes, within window.
    for failure_status, i in zip(
        ["Failed", "QueryNotTraversable", "UnsupportedConstraint"],
        range(3),
        strict=True,
    ):
        created = now - timedelta(minutes=5 * (i + 1))
        duration = timedelta(seconds=(i + 1) * 2)
        docs.append(
            _status_doc(
                status=failure_status,
                submitter="bob",
                tiers=[1],
                created=created,
                completed=created + duration,
                results=0,
            )
        )

    # Two in-flight jobs (one each tier).
    docs.extend(
        _status_doc(
            status="Running",
            submitter="alice" if tier == 0 else "bob",
            tiers=[tier],
            created=now - timedelta(minutes=2),
            completed=None,
            results=0,
        )
        for tier in (0, 1)
    )

    # One older completed job way outside the typical 1-hour window.
    old = now - timedelta(days=7)
    docs.append(
        _status_doc(
            status="Complete",
            submitter="alice",
            tiers=[0],
            created=old,
            completed=old + timedelta(seconds=30),
            results=100,
        )
    )

    await collection.bulk_write([InsertOne(d) for d in docs])  # pyright: ignore[reportUnknownMemberType]


def _status_doc(  # noqa: PLR0913
    *,
    status: str,
    submitter: str,
    tiers: list[int],
    created: datetime,
    completed: datetime | None,
    results: int,
) -> dict[str, Any]:
    """Construct a single synthetic job_status document."""
    doc: dict[str, Any] = {
        "job_id": uuid.uuid4().hex,
        "status": status,
        "submitter": submitter,
        "data_tiers": tiers,
        "is_async": False,
        "qnodes": 2,
        "qedges": 1,
        "qpaths": 0,
        "job_timeout": {"0": 30.0, "1": 60.0},
        "created": created,
        "touched": created,
    }
    if completed is not None:
        doc["completed"] = completed
        doc["knodes"] = 5
        doc["kedges"] = 3
        doc["aux_graphs"] = 0
        doc["results"] = results
    return doc
