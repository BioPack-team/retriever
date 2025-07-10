# BioPack Retriever

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

```bash
# Get into the Virtual Environment.
# Otherwise, add `uv run` in front of the following commands
source .venv/bin/activate
task dev
```

Shut down the database containers when you're done (warning, wipes the containers):

```bash
task dbs:stop
```

## Configuration

See the [Configuration Documentation](docs/CONFIGURATION.md).
