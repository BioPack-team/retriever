# Outage test proxies

Toggleable HTTP reverse proxies that sit between Retriever and a tier backend,
so you can make a backend fail (and recover) **while the server is running** —
without touching the real backend's availability. Use them to exercise any
behavior that depends on a tier being up or down: fallback routing, `/status`
reporting, cross-process health propagation, query timeouts, degraded modes,
error responses, and so on.

- `outage_proxy.py` — the shared proxy (forward / `error` / `hang` modes + control routes).
- `tier0_proxy.py` — launcher for tier 0 (Gandalf), default port `18080`.
- `tier1_proxy.py` — launcher for tier 1 (Elasticsearch), default port `18081`.

In `pass` mode each proxy forwards transparently to its `--upstream` (the real
backend). A control call flips it to fail; another flips it back.

## Setup

Run a proxy pointed at the **real** backend, and point Retriever's tier at the
proxy instead. Either via env vars:

```sh
uv run python scripts/tier0_proxy.py --upstream https://your-gandalf-host:443
export TIER0__GANDALF__HOST=http://127.0.0.1 TIER0__GANDALF__PORT=18080

uv run python scripts/tier1_proxy.py --upstream http://your-es-host:9200
export TIER1__ELASTICSEARCH__HOST=127.0.0.1 TIER1__ELASTICSEARCH__PORT=18081
```

…or by setting the same fields in `config/config.yaml`. Then start the server
as usual and send whatever traffic your test needs.

## Control the outage

```sh
curl -X POST 'http://127.0.0.1:18080/__outage__/down'            # start failing
curl -X POST 'http://127.0.0.1:18080/__outage__/down?mode=hang'  # start stalling
curl -X POST 'http://127.0.0.1:18080/__outage__/up'              # recover
curl      'http://127.0.0.1:18080/__outage__/state'              # inspect mode + upstream
```

| Mode      | Control                             | Simulates                             |
| --------- | ----------------------------------- | ------------------------------------- |
| `error`   | `POST /__outage__/down` (default)   | backend up but returning 503          |
| `hang`    | `POST /__outage__/down?mode=hang`   | unreachable — stalls past the timeout |
| recover   | `POST /__outage__/up`               | backend healthy again                 |
| (inspect) | `GET /__outage__/state`             | current mode + upstream               |

For a hard connection-refused outage, just Ctrl-C the proxy.

## Notes

- The control routes live under the reserved `/__outage__` prefix so they never
  collide with backend paths; everything else is proxied.
- `--hang-seconds` (default 8) sets how long `hang` mode stalls; keep it above
  the driver's connect timeout (5s) so the outage is actually detected.
- Cross-process behavior (e.g. health propagation) is only observable with more
  than one worker, so set `workers` in config when that's what you're testing.
