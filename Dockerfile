# Build via compose: docker compose build
# Build manual: docker build --rm --force-rm --compress -t biopack-team/retriever .
FROM python:3.13-alpine

# Ensure requirements
RUN apk add --no-cache git
# Build requirements
RUN apk add --no-cache rust cargo g++ file make
RUN pip install --upgrade pip

RUN adduser -D python
USER python

WORKDIR /usr/src/app

COPY --chown=python:python . .

RUN pip install --user uv
ENV PATH="/home/python/.local/bin:${PATH}"
RUN uv sync --locked --no-dev

CMD [ ".venv/bin/python3", "./src/retriever/__main__.py" ]
