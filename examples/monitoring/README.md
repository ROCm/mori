# UMBP monitoring

A Prometheus scrape fragment and the Grafana dashboards for a UMBP deployment.
Point Grafana at `grafana/dashboards/` and Prometheus at
`umbp_master.scrape.yml`; `example/docker-compose.yml` is a working stack that
does both.

## The dashboards

| Dashboard | Covers |
| --- | --- |
| `umbp_backends.json` | **Every storage medium and transfer engine**, in shared panels |
| `umbp_data_rate_bandwidth.json` | Per-client inbound/outbound data rate and BatchPut/BatchGet bandwidth |
| `umbp_rpc_call_rates.json` | Client capacity, routing and batch API call rates |
| `umbp_master_client_rpc_latency.json` | Master RPC latency, QPS and error rate by RPC |
| `umbp_external_kv.json` | The external-KV index: report/revoke/match and live block counts |

The last four are backend-agnostic — they measure the client, the master and
the KV index, which look the same whatever medium is underneath — and are
unchanged.

## One dashboard for every medium

`umbp_backends.json` is the one that got merged. It replaced the per-medium
dashboards (`umbp_ssd_tier.json`, and whatever the next medium would have
needed): every storage panel groups by `tier`, and every transfer panel by
`engine`, so DRAM, HBM and SSD appear as series in the same panel and can be
read against each other. There is deliberately **no dashboard per medium**.

That is a property of how the metrics are emitted, not a convention this file
asks you to follow. A medium does not name its own metrics: `InstrumentedBackend`
wraps every `MediumBackend` and derives the generic series from the interface
calls, and `CompositeTransferEngine` does the same for every engine it
dispatches to. **A backend or engine added to the code shows up in these panels
with no dashboard change and no metrics code of its own.**

The one thing a component publishes for itself is state its interface cannot
show from outside — what a device actually did, how full an internal arena is.
Those ride generic names with the specifics in a label
(`mori_umbp_backend_medium_events_total{tier="SSD", event="single_flight_dup"}`),
so they land in the existing "medium-internal" panels rather than needing new
ones. See `src/umbp/include/umbp/distributed/metrics/component_metrics.h` for
the vocabulary and the rule.

## Editing the dashboards

All five are plain Grafana JSON, edited the way Grafana dashboards normally
are: change the panel in the Grafana UI, export the dashboard, commit the JSON.

When you add a panel to `umbp_backends.json`, write the query against
**labels** — `sum by (tier) (...)`, `sum by (engine, direction) (...)` — never
against a metric name or a pinned matcher that spells one medium or one
transport (`mori_umbp_ssd_read_total`, `tier="SSD"`). A panel written the other
way stops covering the system the day someone adds a backend, which is exactly
the failure this dashboard replaced.

## Template variables

`umbp_backends.json` uses `node` and `tier`, populated from
`mori_umbp_client_capacity_total_bytes` — a metric every live deployment
publishes, so the pickers fill in as soon as one client registers.
