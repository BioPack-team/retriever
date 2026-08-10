"""Outage proxy for the tier 0 (Gandalf) backend.

Run it pointed at the real Gandalf endpoint, then point Retriever's tier 0 at
the proxy:

    uv run python scripts/tier0_proxy.py --upstream https://your-gandalf-host:443

    # In Retriever's environment (the proxy defaults to port 18080):
    export TIER0__GANDALF__HOST=http://127.0.0.1
    export TIER0__GANDALF__PORT=18080

Then start the server and, mid-run, toggle the backend to test how Retriever
behaves while tier 0 is down and when it recovers:

    curl -X POST 'http://127.0.0.1:18080/__outage__/down'   # gandalf goes down
    curl -X POST 'http://127.0.0.1:18080/__outage__/up'     # gandalf recovers

See scripts/README.md for modes and the full flow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from outage_proxy import run  # noqa: E402

DEFAULT_PORT = 18080


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        required=True,
        help="Base URL of the REAL Gandalf endpoint, e.g. https://your-gandalf-host:443",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Listen host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Listen port.")
    parser.add_argument(
        "--hang-seconds",
        type=float,
        default=8.0,
        help="Seconds to stall in `hang` mode (above the tier-0 connect timeout, 5s).",
    )
    args = parser.parse_args()
    run(
        name="tier0-gandalf",
        upstream=args.upstream,
        host=args.host,
        port=args.port,
        hang_seconds=args.hang_seconds,
    )


if __name__ == "__main__":
    main()
