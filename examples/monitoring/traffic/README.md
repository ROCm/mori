# MORI traffic dashboards: IO and EP in Grafana

Grafana dashboards for the traffic metrics every MORI process exports (`mori_io_*`,
`mori_ep_*`, `mori_gpu_rdma_*`), and a Prometheus + VictoriaMetrics + Grafana stack to run them.

| Dashboard | Shows |
| --- | --- |
| **MORI traffic overview: IO and EP** | IO and EP on the same links and time axis (do they take turns or overlap), per-GPU rates, GPU-to-GPU (XGMI) and NIC-to-NIC (RDMA) mesh tables, EP bytes / time per call, IO bytes / time per transfer |
| **MORI EP traffic** | dispatch / combine kernel bandwidth, GPU-to-GPU matrix, per-GPU XGMI and per-NIC RDMA |
| **MORI IO traffic** | throughput, latency, failures, GPU-to-GPU matrix, per-GPU and per-NIC traffic by backend |

![Overview](images/overview.png)

## Start

1. Run the jobs with metrics on:

   ```
   export MORI_METRICS_PORT=9400                  # /metrics on 9400 + local rank
   export MORI_METRICS_PORT_DIR=/tmp/mori-metrics # each process writes a Prometheus target file here
   ```

2. Start the stack on the host that shows the dashboards:

   ```
   ./traffic.sh up                                          # Prometheus :9090, Grafana :3000, data in ./data
   TRAFFIC_TARGETS_DIR=/shared/mori-metrics ./traffic.sh up # the jobs' MORI_METRICS_PORT_DIR
   TRAFFIC_TARGETS=node-a:9400,node-a:9401 ./traffic.sh up  # or a fixed list
   ./traffic.sh down                                        # stop; data is kept
   ```

   Everything listens on 127.0.0.1 (no login). From another machine:
   `ssh -N -L 3000:127.0.0.1:3000 <host>`, then open `http://localhost:3000`.
   Ports: `TRAFFIC_PROM_PORT`, `TRAFFIC_GRAFANA_PORT`, `TRAFFIC_VM_PORT`, `TRAFFIC_GATEWAY_PORT`.

3. Check `http://127.0.0.1:9090/targets`: every MORI process is one target; only `UP` ones show.

## Environment variables

Read when a process creates its first IO engine or EP op.

| Variable | Meaning |
| --- | --- |
| `MORI_METRICS_PORT` | Metrics on; `/metrics` on this port + local rank (next free port if taken) |
| `MORI_METRICS_ENABLE=1` | Metrics on without a port (`mori.metrics.snapshot()`); `0` forces off |
| `MORI_METRICS_PORT_DIR` | Directory for each process's Prometheus `file_sd` target file |
| `MORI_METRICS_ADDR` | Address to listen on (default all interfaces) |
| `MORI_METRICS_HOST_ID` | `host` label instead of the hostname |
| `MORI_METRICS_SAMPLE_EP`, `_IO` | Count one call in N, N-fold (default 1 = all, 0 = none) |
| `MORI_METRICS_INTERVAL_MS` | How often counters are read (default 1000; 1 with a push URL) |
| `MORI_METRICS_PUSH_URL` | VictoriaMetrics URL: push every sample, millisecond resolution |

Building with `BUILD_METRICS=OFF` removes the counting code entirely. Cost and accuracy are
measured in [`docs/METRICS_PERF.md`](../../../docs/METRICS_PERF.md).

## Data sources

| Data source | Resolution | How |
| --- | --- | --- |
| `mori-traffic` | one point per scrape (default 5 s) | Prometheus scrapes each process's `/metrics` |
| `mori-traffic-ms` | 1 ms | processes push to VictoriaMetrics (`MORI_METRICS_PUSH_URL=http://<host>:8429`) |

For millisecond data, pick `mori-traffic-ms` and zoom to a few seconds. Points arrive once a
second, so end the time range a couple of seconds back. Pushing costs about 7–11 % of a core
and 0.5–1 MB/s per process; put VictoriaMetrics close to the jobs.

## Switches on the overview

The first two rows are buttons, not charts. They act on every running process and need the
gateway `traffic.sh up` starts ([`sampling_gateway.py`](sampling_gateway.py)). The button in
effect has a ring.

- **Metrics sampling** (`1/1`, `1/8`, `1/64`, `1/100`, `off`, for EP and for IO): count one call
  in N, N-fold, so the charts keep the same scale and the counting costs less. Same as
  `./traffic.sh sampling ep=8 io=64`, `POST /sampling?ep=8` on a process, or
  `mori.metrics.set_sampling(ep=8)`. All ranks sample the same dispatches. Per-NIC RDMA bytes are
  never sampled.
- **Prometheus scrape interval** (`1s`, `2s`, `5s`, `10s`): rewrites `scrape_interval` and
  reloads Prometheus. Shorter = finer `mori-traffic` charts, more load. Does not affect
  `mori-traffic-ms`.

Open Grafana at the address it knows itself by (`localhost:<port>` by default, or
`TRAFFIC_GRAFANA_URL`), or the buttons fail. Anyone who can open the dashboard can change them.

## Reading the dashboards

- **Payload, not link bytes.** RDMA per NIC matches the NIC's own counter (exact on AINIC, about
  1 % below on InfiniBand: headers). XGMI hardware counters (`xgmi-netmon`) read 1.1–1.6× the
  payload depending on the traffic pattern.
- **Rate window** (overview): how far back each chart point averages. `auto` = 20 s on
  `mori-traffic`; `1ms`–`500ms` only make sense on `mori-traffic-ms`. A rate needs two samples
  in its window. **Average over**: the window of the tiles and mesh tables.
- **EP kernel bandwidth vs. payload over wall time**: kernel bandwidth (bytes / device-measured
  kernel time) is what an EP benchmark reports and does not move with the call rate; payload
  over wall time rises and falls with how often EP runs.
- **Mesh tables**: rows send, columns receive. XGMI: one table per host, GPU by GPU. RDMA: one
  table per host pair, NIC by NIC. Each cell is IO + EP.
- An IO transfer is counted once, by the process that started it, with both ends in its labels.
- EP inter-node counts a token once per hop (RDMA to the other node, then XGMI).
- Not counted yet: EP AsyncLL, the gfx125x intra-node kernels, KIWI (only their RDMA bytes).

## Your own Prometheus / Grafana

Copy the `mori_traffic` job from [`prometheus.yml`](prometheus.yml) and import the JSON files in
[`grafana/dashboards/`](grafana/dashboards/) (keep the `uid`s: the dashboards link to each
other). Set the data source's scrape interval to match Prometheus.

## Known limits

- A process killed by a signal leaves its target file; that target shows down until it is deleted.
- Containers with the same hostname writing to a shared target directory need different
  `MORI_METRICS_HOST_ID`s.
- Custom target files must not add labels named `rank`, `gpu` or `host`.
- A process that exits takes its last few seconds with it: whole-run totals read slightly low.
- GPU colours are fixed for GPU 0–7; matrix rows sort as text (GPU10 before GPU2).
