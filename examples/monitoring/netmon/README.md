# netmon dashboards: ainic-netmon recordings in Grafana

VictoriaMetrics + Grafana with two dashboards for recordings made by
[`ainic-netmon`](../../../tools/netmon/README.md):

- **netmon overview**: one experiment, every node and NIC. A table of the NICs (host, RDMA
  device name such as `ionic_0 (benic1p1)`, link rate, average send and receive over the
  selected time; click a NIC to open its detail page) and one small RDMA traffic chart per NIC.
- **netmon NIC detail**: one NIC: RDMA traffic, all port traffic, retransmissions and
  congestion notifications. Collapsed at the bottom: individual observations, PFC pause
  counters (meaning not yet validated), and recording diagnostics for troubleshooting the
  monitor itself (gaps, sampling quality, the offline report's fixed-window summary).

How to read the traffic charts: **blue = send, orange = receive; solid = average, dashed =
observed peak; gaps = data unavailable.** Pick the experiment, then the host, then the NIC.

Data gets there in two ways: live while recording (`ainic-netmon record --push`), and after
the fact with exact numbers (`ainic-netmon-import`).

## Start

```
./up.sh                    # data in ./data; or ./up.sh /scratch/netmon
NETMON_VM_PORT=18428 NETMON_GRAFANA_PORT=13000 ./up.sh   # other ports
```

`up.sh` writes `.env` (your uid/gid, the data directory, ports, retention) and runs
`docker compose up -d`. Both containers run as you and listen on 127.0.0.1 only; Grafana
allows anonymous admin access, so do not expose it. `./down.sh` stops them and keeps the data.

## Live view while recording

```
ainic-netmon record -o /shared/exp42/{host} --push http://127.0.0.1:8428
```

- The recorder prints two dashboard links (last 5 minutes, refreshing every 2 s). Each node
  pushes its own NICs; nodes recording into the same experiment appear together.
- The recording itself is never affected: if VictoriaMetrics is slow or goes away, points are
  dropped and counted (`push` in `meta.json`, warnings on stderr), and pushing resumes when it
  is back. The recorder refuses to start if VictoriaMetrics is unreachable, or if this host
  already pushed to this experiment.
- Live numbers are an online approximation of the offline rules (slow samples and the
  low-resolution threshold are judged on the last 1024 samples instead of the whole run; in
  tests 99.95 % of the points are identical). In a window where a push failed, the average may
  be off (part of a request can be lost). The offline report summary stays empty while live. When the
  recording ends, import it with `--replace` to replace the live data with the exact numbers.

### Several nodes

Every node pushes to the same VictoriaMetrics; run the stack on one of them (node A here).
VictoriaMetrics has no authentication, and an open port also accepts deletes, so never expose
it beyond the nodes that push.

- **Internal network (recommended):** on node A, `NETMON_VM_LISTEN=<A's internal IP> ./up.sh`.
  VictoriaMetrics then also listens on that address; Grafana stays on 127.0.0.1. Allow only the
  pushing nodes to reach the port on that interface: first a rule that drops everyone, then
  one accept rule per pushing node inserted above it, for example (internal interface `eth1`)

  ```
  iptables -I INPUT 1 -i eth1 -p tcp --dport 8428 -j DROP
  iptables -I INPUT 1 -i eth1 -p tcp --dport 8428 -s <node B> -j ACCEPT
  iptables -I INPUT 1 -i eth1 -p tcp --dport 8428 -s <node C> -j ACCEPT
  ```

  (or an equivalent firewall / security-group rule). Other nodes record with
  `--push http://<A's internal IP>:8428`.
- **No firewall control, or no direct route:** keep VictoriaMetrics on 127.0.0.1 and give each
  other node a tunnel to it, e.g. from a machine that can reach both:
  `ssh -N -L 18428:127.0.0.1:8428 nodeA` plus `ssh -N -R 8428:127.0.0.1:18428 nodeB`, then on
  node B `--push http://127.0.0.1:8428`.

To look at the dashboards from your desktop: `ssh -L 3000:127.0.0.1:3000 nodeA`, then open
http://127.0.0.1:3000.

## Import and look

```
ainic-netmon-import /shared/exp42/nodeA /shared/exp42/nodeB
```

- The experiment name defaults to the recordings' parent directory (`exp42`); `--experiment`
  sets it.
- The importer prints the two dashboard URLs for the experiment's time range. The smallest
  time bin of the charts (3× the sampling interval, e.g. 6 ms for 2 ms sampling) is taken
  from the data (hidden variable `min_step`).
- Experiments imported by an older importer have no RDMA device names: their NICs are listed
  as `host/netdev` until re-imported with `--replace`.
- An experiment is imported as a whole (its fixed window, `min_step` and time anchor depend
  on every recording). Re-importing needs `--replace` with all of its recordings; it deletes
  only `netmon_*` series of that experiment. Recordings that do not overlap in time are
  refused: import them as separate experiments. `--dry-run` shows what would be written.
- `ainic-netmon-import verify RUN_DIR...` re-checks VictoriaMetrics and every dashboard query
  (through Grafana) against the offline report.

## What the panels show (and do not)

- Rates use the offline report's rules: sample time `t0 + f·rtt`, requests slower than 1.5×
  the median are skipped, counter resets and wall-clock steps cut the series. Each
  observation is the rate between two counter reads (about 2 ms apart).
- Per time bin `(t − step, t]`: the average (solid) is Σ increase / Σ duration of the valid
  observations ending in the bin, the peak (dashed) is the highest of them; both include the
  longer observations around sampling stalls. A single observation has timing jitter and can
  slightly exceed the link rate. A bin without a valid observation has no point: lines break
  there; nothing is filled with zeros or carried forward.
- The charts never use a bin smaller than 3× the sampling interval; for single observations
  open "Inspect individual observations" on the NIC detail page and zoom to 1 ms bins.
- The overview table averages the observations that end in the selected time range (a long
  observation crossing its start is counted whole).
- Only NIC counters are shown; nothing about the workload running over the NICs is
  inferred.
- The offline report summary (recording diagnostics) is for the experiment's fixed common
  window; it does not change with zoom.

## Notes

- Ports default to VictoriaMetrics 8428 and Grafana 3000.
- VictoriaMetrics keeps data for `NETMON_VM_RETENTION` (default 100y): imported recordings
  are historical and would otherwise be dropped silently.
- The dashboards are generated by `tools/netmon/report/ainic_netmon_dashboards.py`; edit
  that and regenerate instead of editing the JSON.
