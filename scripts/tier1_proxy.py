"""Outage proxy for the tier 1 (Elasticsearch) backend.

Run it pointed at the real Elasticsearch, then point Retriever's tier 1 at the
proxy:

    uv run python scripts/tier1_proxy.py --upstream http://your-es-host:9200

    # In Retriever's environment (the proxy defaults to port 18081):
    export TIER1__ELASTICSEARCH__HOST=127.0.0.1
    export TIER1__ELASTICSEARCH__PORT=18081

Then start the server and, mid-run, toggle the backend to test how Retriever
behaves while tier 1 is down and when it recovers:

    curl -X POST 'http://127.0.0.1:18081/__outage__/down'   # ES goes down
    curl -X POST 'http://127.0.0.1:18081/__outage__/up'     # ES recovers

See scripts/README.md for modes and the full flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from outage_proxy import run  # noqa: E402

DEFAULT_PORT = 18081


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        required=True,
        help="Base URL of the REAL Elasticsearch, e.g. http://your-es-host:9200",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Listen host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Listen port.")
    parser.add_argument(
        "--hang-seconds",
        type=float,
        default=8.0,
        help="Seconds to stall in `hang` mode (above the tier-1 connect timeout, 5s).",
    )
    args = parser.parse_args()
    run(
        name="tier1-elasticsearch",
        upstream=args.upstream,
        host=args.host,
        port=args.port,
        hang_seconds=args.hang_seconds,
    )


if __name__ == "__main__":
    main()
