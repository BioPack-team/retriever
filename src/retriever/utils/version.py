"""Shared lookup for the running build's git commit + branch.

Two callers (status.py for /status, server.py for /config) used to each
spawn their own gitpython lookup on every request. Now both go through
`get_version()`, which resolves in this order:

  1. `RETRIEVER_GIT_COMMIT` / `RETRIEVER_GIT_BRANCH` env vars — an
     explicit override for deployments that bake version info via env.
  2. `_version_commit.txt` / `_version_branch.txt` sitting next to this
     module — the Dockerfile writes these at build time via `git
     rev-parse` so shipped images report the real SHA out of the box.
  3. A runtime gitpython lookup — used for editable / `uv run` dev
     installs where the source tree's `.git` is reachable.
  4. Fallback `("unknown", None)`, logged once per worker.

Result is cached so the answer is computed once per process.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import git
from loguru import logger

_PKG_DIR = Path(__file__).parent.parent
_COMMIT_FILE = _PKG_DIR / "_version_commit.txt"
_BRANCH_FILE = _PKG_DIR / "_version_branch.txt"


def _read_baked() -> tuple[str, str | None] | None:
    """Read `(sha, branch)` from the Dockerfile-baked version files.

    Returns None when the commit file is missing or empty (e.g. local
    `uv run` outside a container, or a build where `git rev-parse`
    couldn't read the repo).
    """
    if not _COMMIT_FILE.exists():
        return None
    sha = _COMMIT_FILE.read_text().strip()
    if not sha:
        return None
    branch_raw = _BRANCH_FILE.read_text().strip() if _BRANCH_FILE.exists() else ""
    # `git rev-parse --abbrev-ref HEAD` returns "HEAD" on detached HEAD;
    # treat that (and empty) as "no branch".
    branch = None if branch_raw in ("", "HEAD") else branch_raw
    return sha, branch


@functools.cache
def get_version() -> tuple[str, str | None]:
    """Return `(git_commit, git_branch)` for the running build."""
    env_sha = os.environ.get("RETRIEVER_GIT_COMMIT")
    if env_sha:
        env_branch = os.environ.get("RETRIEVER_GIT_BRANCH") or None
        return env_sha, env_branch

    baked = _read_baked()
    if baked is not None:
        return baked

    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        try:
            branch = repo.active_branch.name
        except TypeError:
            # Detached HEAD — no active branch.
            branch = None
        return sha, branch
    except Exception:
        logger.info(
            "Git metadata unavailable; reporting version as 'unknown'.",
            no_mongo_log=True,
        )
        return "unknown", None
