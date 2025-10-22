# BioPack Retriever

[![Code Checks](https://github.com/BioPack-team/retriever/actions/workflows/checks.yml/badge.svg)](https://github.com/BioPack-team/retriever/actions/workflows/checks.yml)
[![codecov](https://codecov.io/gh/BioPack-team/retriever/graph/badge.svg?token=GGUM6QAQKJ)](https://codecov.io/gh/BioPack-team/retriever)

Intermediary between Knowledge Providers and [Shepherd](https://github.com/BioPack-team/shepherd), deduplicating subquery operations, cache layer, centralized normalization calls.

## Installation

For a more detailed overview, see the [Installation Documentation](./docs/INSTALLATION.md).

> [!NOTE]
> Requires [Docker](https://www.docker.com/get-started/) and Python ~3.13

Quick setup for local workspace:

```bash
git clone https://github.com/BioPack-team/retriever
cd retriever
pip install uv # If you don't have uv installed
uv sync
```

## Usage

Quick start:

> [!NOTE]
> Retriever uses external database backends for its data. These backends do not currently have public access.
> If you wish to test Retriever, you'll either need to be granted access (Translator devs only),
> or stand up your own backends.

```bash
# Get into the Virtual Environment.
# Otherwise, add `uv run` in front of the following commands
source .venv/bin/activate
task start # or `task dev` if you want some useful options (tracelogs, single worker, etc.)
```

Shut down the database containers when you're done (warning, wipes the containers):

```bash
task dbs:stop
```

## Configuration

See the [Configuration Documentation](docs/CONFIGURATION.md).
