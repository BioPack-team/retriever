"""Live tests for GridFS spillover of oversized job-doc blobs.

Run with: `pytest -m live tests/test_utils_mongo_gridfs_live.py`. Requires a
reachable Mongo (see the docker compose container). Uses the throwaway test DB
from the shared `test_mongo` fixture, which also redirects the GridFS bucket,
so everything is dropped on teardown.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest
from utils.mongo_fixtures import (
    test_mongo,  # noqa: F401  pyright:ignore[reportImplicitRelativeImport]  # fixture import
)

from retriever.config.general import CONFIG
from retriever.utils.mongo import (
    GRIDFS_INLINE_LIMIT,
    JOB_DOCS_FS_BUCKET,
    MongoClient,
    MongoQueue,
    QueryState,
    ResponseState,
)

pytestmark = pytest.mark.live


def _response_state(job_id: str, payload: bytes) -> ResponseState:
    """A minimal ResponseState carrying `payload` as the stored doc blob."""
    return ResponseState(
        job_id=job_id,
        response=payload,
        knodes=1,
        kedges=1,
        aux_graphs=0,
        results=1,
        status="Success",
    )


async def _write(state: ResponseState) -> None:
    """Persist one state through the queue path that decides inline vs GridFS."""
    await MongoQueue().job_state([state])


@pytest.mark.asyncio
async def test_large_blob_round_trips_via_gridfs(test_mongo: MongoClient) -> None:  # noqa: F811
    """A blob over the inline limit is spilled to GridFS and hydrated back intact."""
    job_id = uuid.uuid4().hex
    payload = b"\xa5" * (GRIDFS_INLINE_LIMIT + 4096)
    await _write(_response_state(job_id, payload))

    _, docs = test_mongo.get_job_collection()
    raw = await docs.find_one({"job_id": job_id})
    assert raw is not None
    assert raw.get("doc") is None  # not stored inline
    assert raw.get("doc_ref") is not None  # spilled to GridFS
    assert raw.get("doc_size") == len(payload)

    files = test_mongo._doc_blob_files()
    assert await files.count_documents({"metadata.job_id": job_id}) == 1

    job = await test_mongo.get_job_doc(job_id)
    assert job is not None
    assert job["doc"] == payload  # hydrated from GridFS, byte-for-byte


@pytest.mark.asyncio
async def test_small_blob_stored_inline(test_mongo: MongoClient) -> None:  # noqa: F811
    """A blob under the inline limit stays in the document; no GridFS file."""
    job_id = uuid.uuid4().hex
    payload = b"small-response-blob"
    await _write(_response_state(job_id, payload))

    _, docs = test_mongo.get_job_collection()
    raw = await docs.find_one({"job_id": job_id})
    assert raw is not None
    assert raw.get("doc") == payload
    assert raw.get("doc_ref") is None

    files = test_mongo._doc_blob_files()
    assert await files.count_documents({"metadata.job_id": job_id}) == 0

    job = await test_mongo.get_job_doc(job_id)
    assert job is not None
    assert job["doc"] == payload


@pytest.mark.asyncio
async def test_overwrite_leaves_single_gridfs_file(test_mongo: MongoClient) -> None:  # noqa: F811
    """Re-spilling a job's blob drops the prior file, leaving exactly one."""
    job_id = uuid.uuid4().hex
    first = b"\x01" * (GRIDFS_INLINE_LIMIT + 100)
    second = b"\x02" * (GRIDFS_INLINE_LIMIT + 200)
    await _write(_response_state(job_id, first))
    await _write(_response_state(job_id, second))

    files = test_mongo._doc_blob_files()
    assert await files.count_documents({"metadata.job_id": job_id}) == 1

    job = await test_mongo.get_job_doc(job_id)
    assert job is not None
    assert job["doc"] == second


@pytest.mark.asyncio
async def test_reaper_deletes_orphaned_blob(
    test_mongo: MongoClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reaper drops a blob (and its chunks) whose job_docs doc is gone."""
    # Drop the grace window so a just-uploaded file is immediately eligible.
    monkeypatch.setattr("retriever.utils.mongo.GRIDFS_REAP_GRACE", timedelta(0))

    alive = uuid.uuid4().hex
    orphan = uuid.uuid4().hex
    await _write(_response_state(alive, b"\x03" * (GRIDFS_INLINE_LIMIT + 1)))
    await _write(_response_state(orphan, b"\x04" * (GRIDFS_INLINE_LIMIT + 1)))

    # Simulate the orphan's job_docs document being TTL-expired.
    _, docs = test_mongo.get_job_collection()
    await docs.delete_one({"job_id": orphan})

    assert await test_mongo.reap_orphaned_doc_blobs() == 1

    files = test_mongo._doc_blob_files()
    chunks = test_mongo._persist_db().get_collection(f"{JOB_DOCS_FS_BUCKET}.chunks")
    assert await files.count_documents({"metadata.job_id": orphan}) == 0
    assert await files.count_documents({"metadata.job_id": alive}) == 1

    # The orphan's chunks are gone too; only the surviving file's remain.
    alive_file = await files.find_one({"metadata.job_id": alive})
    assert alive_file is not None
    assert await chunks.count_documents({"files_id": {"$ne": alive_file["_id"]}}) == 0
    assert await chunks.count_documents({"files_id": alive_file["_id"]}) > 0


@pytest.mark.asyncio
async def test_reaper_streams_across_batches(
    test_mongo: MongoClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reaping is correct when candidates span multiple batches (batch size 1)."""
    monkeypatch.setattr("retriever.utils.mongo.GRIDFS_REAP_GRACE", timedelta(0))
    monkeypatch.setattr("retriever.utils.mongo.GRIDFS_REAP_BATCH", 1)

    alive = uuid.uuid4().hex
    orphans = [uuid.uuid4().hex for _ in range(2)]
    for job_id in (alive, *orphans):
        await _write(_response_state(job_id, b"\x05" * (GRIDFS_INLINE_LIMIT + 1)))

    _, docs = test_mongo.get_job_collection()
    await docs.delete_many({"job_id": {"$in": orphans}})

    assert await test_mongo.reap_orphaned_doc_blobs() == 2

    files = test_mongo._doc_blob_files()
    assert await files.count_documents({"metadata.job_id": {"$in": orphans}}) == 0
    assert await files.count_documents({"metadata.job_id": alive}) == 1


@pytest.mark.asyncio
async def test_blob_at_cap_is_delivered_but_not_stored(
    test_mongo: MongoClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blob at/above the cap leaves a terminal status but no stored body."""
    cap = 512 * 1024
    monkeypatch.setattr(CONFIG.mongo, "max_stored_doc_bytes", cap)

    job_id = uuid.uuid4().hex
    await _write(_response_state(job_id, b"\x06" * (cap + 1024)))

    status, docs = test_mongo.get_job_collection()
    # Status is still recorded, so the job reads back as a completed job.
    status_doc = await status.find_one({"job_id": job_id})
    assert status_doc is not None
    assert status_doc["status"] == "Success"

    # The doc carries no body in any form, and nothing was spilled to GridFS.
    raw = await docs.find_one({"job_id": job_id})
    assert raw is not None
    assert raw.get("doc") is None
    assert raw.get("doc_ref") is None
    assert raw.get("doc_size") is None
    assert (
        await test_mongo._doc_blob_files().count_documents({"metadata.job_id": job_id})
        == 0
    )

    job = await test_mongo.get_job_doc(job_id)
    assert job is not None
    assert job.get("doc") is None


@pytest.mark.asyncio
async def test_cap_clears_a_previously_stored_body(
    test_mongo: MongoClient,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An over-cap rewrite drops the body stored by an earlier under-cap write."""
    job_id = uuid.uuid4().hex
    # First write is small enough to store inline.
    await _write(_response_state(job_id, b"first-small-body"))

    _, docs = test_mongo.get_job_collection()
    assert (await docs.find_one({"job_id": job_id}) or {}).get("doc") is not None

    # Second write is over the cap, so the prior body must be cleared.
    cap = 512 * 1024
    monkeypatch.setattr(CONFIG.mongo, "max_stored_doc_bytes", cap)
    await _write(_response_state(job_id, b"\x07" * (cap + 1024)))

    raw = await docs.find_one({"job_id": job_id})
    assert raw is not None
    assert raw.get("doc") is None
    assert raw.get("doc_ref") is None


@pytest.mark.asyncio
async def test_dehydrated_response_stores_state_but_no_body(
    test_mongo: MongoClient,  # noqa: F811
) -> None:
    """Dehydrated queries persist their state (counts/status) but never a body."""
    job_id = uuid.uuid4().hex
    # An earlier inline write (e.g. the initial query state) leaves a body...
    await _write(_response_state(job_id, b"earlier-inline-body"))
    status, docs = test_mongo.get_job_collection()
    assert (await docs.find_one({"job_id": job_id}) or {}).get("doc") is not None

    # ...which the dehydrated response write must clear, keeping state only.
    await MongoQueue().job_state(
        [
            ResponseState(
                job_id=job_id,
                knodes=3,
                kedges=2,
                aux_graphs=0,
                results=5,
                status="Success",
                dehydrated=True,
            )
        ]
    )

    status_doc = await status.find_one({"job_id": job_id})
    assert status_doc is not None
    assert status_doc["status"] == "Success"
    assert status_doc.get("dehydrated") is True
    assert status_doc.get("results") == 5  # state is recorded

    raw = await docs.find_one({"job_id": job_id})
    assert raw is not None
    assert raw.get("doc") is None
    assert raw.get("doc_ref") is None
    assert (
        await test_mongo._doc_blob_files().count_documents({"metadata.job_id": job_id})
        == 0
    )

    job = await test_mongo.get_job_doc(job_id)
    assert job is not None
    assert job.get("doc") is None


@pytest.mark.asyncio
async def test_dehydrated_flag_recorded_at_enqueue(test_mongo: MongoClient) -> None:  # noqa: F811
    """The initial query write records `dehydrated` and still stores the query body."""
    job_id = uuid.uuid4().hex
    await MongoQueue().job_state(
        [
            QueryState(
                job_id=job_id,
                query=b"compressed-query-bytes",
                job_timeout=30.0,
                submitter="tester",
                data_tier=0,
                is_async=True,
                qnodes=2,
                qedges=1,
                qpaths=0,
                status="Running",
                worker_pid=1234,
                worker_started_at=datetime.now().astimezone(),
                dehydrated=True,
            )
        ]
    )

    status, docs = test_mongo.get_job_collection()
    status_doc = await status.find_one({"job_id": job_id})
    assert status_doc is not None
    assert status_doc.get("dehydrated") is True  # marker survives even if abandoned

    # Dehydrated only skips response bodies; the query itself is still stored.
    raw = await docs.find_one({"job_id": job_id})
    assert raw is not None
    assert raw.get("doc") == b"compressed-query-bytes"
