# Installation

Retriever can be used locally either in a Docker container, or in a local workspace for development and contributing.

Either way, you'll need [Docker](https://www.docker.com/get-started/) to run local [Dragonfly](https://www.dragonflydb.io/docs) and [MongoDB](https://www.mongodb.com/products/self-managed/community-edition) containers used by Retriever. Follow their [installation instructions](https://docs.docker.com/desktop/) (or install [Colima](https://github.com/abiosoft/colima) as an alternative) to get set up.

## Docker

### Build

Retriever can be built via its [Dockerfile](https://github.com/BioPack-team/retriever/blob/a3049af6bd33e0dc2f45f3a5809117dcc9f3cec8/Dockerfile) either manually or using docker-compose.

Using docker-compose (preferred):

```bash
docker compose build
```

Using the Dockerfile:

```bash
docker build --rm --force-rm --compress -t biopack-team/retriever
```

### Run

Retriever can be run either manually or through docker-compose.

Using docker-compose (preferred):

```bash
docker compose up -d
# Follow the logs if you wish:
docker logs -f retriever
```

Manually:

```bash
docker run \
  -it \
  --name \
  retriever \
  -p 3000:3000 \
  -e REDIS__HOST=host.docker.internal \
  -e MONGO__HOST=host.docker.internal \
  biopack-team/retriever
```

## Local Workspace

Retriever uses [uv](https://docs.astral.sh/uv/) for package/workspace management. See [their documentation](https://docs.astral.sh/uv/getting-started/) for more details. The quickest way to install it is through pip:

```bash
pip install uv
```

`uv` should handle Python versions for you-- Retriever requires Python ~3.12, but ~3.13 is recommended.

From here, you're ready to clone the workspace and get set up:

```bash
git clone https://github.com/BioPack-team/retriever
cd retriever
uv sync
```

And you're ready to start using/developing Retriever! See [Usage](../README.md#Usage) for more.
