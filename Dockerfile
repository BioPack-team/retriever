# Build via compose: docker compose build
# Build manual: docker build --rm --force-rm --compress -t biopack-team/retriever .
FROM python:3.13-alpine

# Ensure requirements
RUN apk add --no-cache git
# Build requirements
RUN apk add --no-cache rust cargo g++ gcc file make python3-dev musl-dev linux-headers
RUN pip install --upgrade pip

RUN adduser -D python
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
