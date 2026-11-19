# netmon dashboards: NIC (and XGMI) recordings in Grafana

One VictoriaMetrics + Grafana for the cluster monitoring tools: NIC counters from
[`ainic-netmon`](../../../tools/netmon/README.md), and other tools that add their dashboard to
`grafana/dashboards/` (see "One stack for several tools" below). The two NIC dashboards:

- **netmon overview**: one experiment, every node and NIC. A table of the NICs (host, RDMA
  device name such as `ionic_0 (benic1p1)`, link rate, average send and receive over the
  selected time; click a NIC to open its detail page) and one small RDMA traffic chart per NIC.
- **netmon NIC detail**: RDMA traffic, all port traffic, retransmissions and congestion
  notifications of one NIC, of several, or of all of them (Host and NIC offer "All"). Every
  chart is drawn once per selected NIC: the traffic charts side by side, the congestion
  charts as one line per NIC. Collapsed at the bottom: individual observations, PFC pause
  counters (meaning not yet validated), and recording diagnostics for troubleshooting the
  monitor itself (gaps, sampling quality, the offline report's fixed-window summary).

How to read the traffic charts: **blue = send, orange = receive; solid = average, dashed =
observed peak; gaps = data unavailable.** Pick the experiment, then the hosts, then the NICs.
With many NICs selected the page is long: charts load as you scroll to them.

The XGMI dashboard of [`xgmi-netmon`](../../../tools/xgmi_netmon/README.md) is a third page in
the same folder: the sent and received bandwidth of every GPU and every XGMI link of one host,
and its PCIe traffic. Grafana's home page is the NIC overview and does not link to it: open
Dashboards, folder `netmon`, `xgmi-netmon`, or `http://127.0.0.1:3000/d/xgmi-netmon`. Its
experiment list shows the experiments that have XGMI data in the selected time range.

Data gets there in two ways: live while recording (`ainic-netmon record --push`), and after
the fact with exact numbers (`ainic-netmon-import`).

## Start

```
./up.sh                    # data in ./data; or ./up.sh /scratch/netmon (local disk)
NETMON_VM_PORT=18428 NETMON_GRAFANA_PORT=13000 ./up.sh   # other ports
```

`up.sh` starts two containers with `docker run`, `netmon-<you>-vm` and `netmon-<you>-grafana`
(so several people can each run one on a shared host, on different ports). Both run as you and
listen on 127.0.0.1 only; Grafana allows anonymous admin access, so do not expose it. Grafana
gets a copy of `grafana/` in the data directory (the docker daemon may not see an NFS home),
refreshed on every start. `./down.sh` stops them and keeps the data.

Moving from the earlier docker-compose version of this stack (`netmon-victoriametrics-1`,
`netmon-grafana-1`): run the new `./down.sh` first (it stops those two containers if they run as
you), then `./up.sh` with the same data directory and ports. `up.sh` refuses to start while that
stack runs or while a port is taken.

### One stack for several tools

The NIC dashboards and the XGMI dashboard of `xgmi-netmon` share this VictoriaMetrics and
Grafana: push and import both tools to the same URL. Their metrics do not collide
(`netmon_*`, `xgmi_*`), and each tool only replaces or refuses its own series. A tool adds its
dashboard as a JSON file in `grafana/dashboards/`, using the data source uid `netmon-vm`; all
of them appear in Grafana's `netmon` folder. For example, with the default ports:

```
ainic-netmon record -o /shared/exp42/{host} --push http://127.0.0.1:8428
xgmi-netmon  record -o /shared/exp42/{host} --push http://127.0.0.1:8428
ainic-netmon-import /shared/exp42/* --url http://127.0.0.1:8428
xgmi-netmon  import /shared/exp42/*     # --url is http://127.0.0.1:8428 by default
```

Both tools can also record into the same directory (`-o /shared/exp42/{host}`, so the default
experiment name is the same): each tool's files start with its own prefix (`nic_*`,
`xgmi_*`), each refuses a directory only if its own files are there, and each importer reads
only its own. `{host}` is the full host name in both (`--host-id` to override).

## Live view while recording

```
ainic-netmon record -o /shared/exp42/{host} --push http://127.0.0.1:8428
```

- The recorder prints two dashboard links (last 5 minutes, refreshing every 2 s). Each node
  pushes its own NICs; nodes recording into the same experiment appear together.
- The recording itself is never affected: if VictoriaMetrics is slow or goes away, points are
  dropped and counted (`push` in `nic_meta.json`, warnings on stderr), and pushing resumes when it
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
- The NIC dashboards are generated by `tools/netmon/report/ainic_netmon_dashboards.py`; edit
  that and regenerate instead of editing the JSON. `xgmi.json` is not generated: it is written
  by hand, edit the JSON itself.
