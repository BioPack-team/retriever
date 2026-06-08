# Build via compose: docker compose build
# Build manual: docker build --rm --force-rm --compress -t biopack-team/retriever .
# DO NOT USE ALPINE, breaks with otel+other deps
FROM python:3.13-slim

# Ensure requirements. git is also used below to bake in the build's commit/branch.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
# No build toolchain needed: every native dependency ships a manylinux (glibc)
# wheel for cp313, so `uv sync` installs prebuilt wheels rather than compiling.
RUN pip install --upgrade pip

RUN useradd --create-home --shell /bin/bash python
USER python

WORKDIR /usr/src/app

COPY --chown=python:python . .

# Bake the running build's git commit + branch into small files
RUN sh -ec '\
        git -C /usr/src/app rev-parse HEAD \
            > /usr/src/app/src/retriever/_version_commit.txt 2>/dev/null || :; \
        git -C /usr/src/app rev-parse --abbrev-ref HEAD \
            > /usr/src/app/src/retriever/_version_branch.txt 2>/dev/null || :; \
        touch /usr/src/app/src/retriever/_version_commit.txt \
              /usr/src/app/src/retriever/_version_branch.txt'

RUN pip install --user uv
ENV PATH="/home/python/.local/bin:${PATH}"
RUN uv sync --locked --no-dev

CMD [ ".venv/bin/python3", "./src/retriever/__main__.py" ]
