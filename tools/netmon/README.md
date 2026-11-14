# ainic-netmon

Millisecond-level counter collector for AMD AINIC (Pensando Salina, `ionic` driver), with an
offline HTML report. A standalone tool: it depends only on libc and Linux kernel headers (the
report needs Python with numpy and plotly), not on the rest of this repository or ROCm.

Typical use: record every NIC of every node for the duration of an experiment, then look at
the per-NIC send/receive rates at 2 ms resolution, together with congestion signals (CNP, ECN,
retransmissions) and, with root, wire bytes and per-priority PFC pause counters.

## Why not sysfs / `ethtool -S`?

Measured on ionic (firmware 1.117.5-a-77):

| Source | Read cost | Refresh | Notes |
|---|---|---|---|
| `/sys/class/net/*/statistics` | 0.2 µs | - | excludes RoCE traffic |
| `ethtool -S` (`SIOCETHTOOL`) | 1.2 µs | **~5 s** for the port (MAC) part | firmware DMAs the port stats every ~5 s |
| `/sys/class/infiniband/*/hw_counters` | 0.9–2.7 ms | ~12.9 ms | ib_core caches values for `lifespan` (10 ms) |
| **RDMA netlink `STAT_GET`** (`rdma` source) | ~1.7 ms | **every request** | unprivileged; 54 RDMA counters |
| **Firmware IPC** (`ipc` source) | ~1.6 ms | **every request** | root; port MAC stats, the channel `nicctl` uses |

So the collector uses RDMA netlink and the firmware IPC channel, sampling each NIC every 2 ms
(about 500 samples/s per source), and ethtool once per second.

## Build

```
cmake -S tools/netmon -B build-netmon -DCMAKE_BUILD_TYPE=Release
cmake --build build-netmon -j
ctest --test-dir build-netmon          # unit tests, no hardware needed
```

The binary `build-netmon/ainic-netmon` links glibc dynamically and libstdc++ statically, so it
can be copied to other hosts and containers (glibc >= the build host's). The Docker image
(`docker/Dockerfile`) builds on Ubuntu 22.04 (glibc 2.35) and also contains the report tool.

## Installed with mori

mori builds this tool by default (`BUILD_NETMON=ON`) and ships it in the wheel, so after
`pip install` of mori:

```
ainic-netmon list                       # or: mori netmon list
ainic-netmon record -o DIR -d 60s       # unprivileged: rdma + ethtool
ainic-netmon record --sudo -o DIR ...   # re-runs the binary with sudo (all sources)
sudo "$(mori netmon --path)" record ... # equivalent, by absolute path
ainic-netmon-report report DIR -o report.html   # or: mori netmon-report ...
```

`sudo ainic-netmon` on its own usually fails because sudo resets `PATH` (pip installs
commands into a virtualenv or `~/.local/bin`). The report needs `numpy` and `plotly` in the
Python environment (`pip install amd_mori[netmon]`). Build mori with `BUILD_NETMON=OFF` to leave the tool out.

## Deployment: three ways, from most to least data

| Where | How | Sources |
|---|---|---|
| Privileged container | `docker run --rm --privileged --network host -v DIR:DIR <image> record ...` | rdma, ipc, ethtool |
| Host with sudo | `sudo ainic-netmon record ...`, or `ainic-netmon record --sudo ...` | rdma, ipc, ethtool |
| Unprivileged (host or normal container) | `ainic-netmon record ...` | rdma, ethtool |

Without root the tool says which sources it skipped and why; the RDMA counters (bytes, packets,
CNP, ECN, retransmissions) are all still recorded. Only the wire bytes and PFC pause counters
need the `ipc` source.

- `--network host` is required in containers (NIC discovery and ethtool use the host network
  namespace). Instead of `--privileged` you can pass the IPC devices with `--device`;
  `ainic-netmon list` prints the exact arguments for the host.
- Output ownership: when running as root, the tool opens the devices, then drops to a normal
  user before creating any file: `--owner UID:GID`, else the user who ran `sudo`, else the
  owner of the nearest existing parent of the output directory. So files written from a
  privileged container to a shared directory belong to you, not to root.
- `--sudo` re-runs the binary by absolute path with sudo (needed because `sudo` resets `PATH`,
  e.g. for pip-installed commands). The tool never escalates on its own.

## Usage

```
ainic-netmon list                    # NICs, PCI / RDMA / IPC mapping, firmware, sources
sudo ainic-netmon verify             # check the IPC reply layout against ethtool (per firmware)
ainic-netmon record -o DIR -d 60s    # record; Ctrl-C or SIGTERM stops early
ainic-netmon record --help
```

Multi-node: run one recorder per node with a shared output path, e.g.

```
ainic-netmon record -o /shared/exp42/{host} -d 120s     # on every node
```

`--tag KEY=VALUE` stores free-form labels in `meta.json`; the tools do not interpret them.

`{host}` expands to the host id (hostname, or `--host-id`). Sampling is aligned to the
wall-clock grid on every node; nodes need NTP/PTP for millisecond alignment across hosts.

The output directory must not exist or be empty (a recording is never overwritten).

Live view in Grafana while recording (VictoriaMetrics + Grafana from
[`examples/monitoring/netmon`](../../examples/monitoring/netmon/README.md)):

```
ainic-netmon record -o /shared/exp42/{host} --push http://127.0.0.1:8428
```

The push runs on its own thread and never slows the samplers or the writer: when
VictoriaMetrics is slow or unreachable, points are dropped and counted (`push` in
`meta.json`). Live values are an online approximation; afterwards
`ainic-netmon-import --replace` replaces them with the offline report's exact numbers.

### Notes and limits

- **IPC** is an undocumented firmware protocol. It is enabled automatically only on firmware
  versions whose layout was verified (`ainic-netmon list` marks others with `*`;
  `--ipc-unverified` overrides). Every reply is validated byte by byte, and a startup check
  compares it with ethtool's (older) copy of the same counters.
- Each `rdma` sample is one firmware admin command (~500/s per NIC at 2 ms). The impact on
  the control path of a running workload (QP creation/modification) has not been measured yet;
  raise `--interval rdma=...` if in doubt.
- The firmware periodically stalls requests: rdma ~100 ms every ~5 s, ipc ~136 ms every 10 s
  per card. Counters are cumulative, so no bytes are lost; resolution drops to one interval
  there (the report marks these intervals).

## Output format

```
DIR/.lock                       created exclusively; pid, host, start time
DIR/meta.json                   status (running|complete|interrupted|failed), host, NIC map,
                                streams (file, source, interval, counter names), degraded
                                sources, run statistics
DIR/<netdev>.<source>.bin       fixed-size records of little-endian u64:
                                [t0_mono_ns, t1_mono_ns, real_ns, real_err_ns, v0 .. v(n-1)]
```

`t0/t1` are `CLOCK_MONOTONIC` around the request; `real` is `CLOCK_REALTIME` read just before
`t0`, and `real_err` is the time it took to read it (an error bound for wall-clock
alignment). A recording killed mid-write may end with a partial record; readers drop it.

```python
import json, numpy as np
m = json.load(open("DIR/meta.json")); s = m["streams"][0]
rec = np.memmap(f"DIR/{s['file']}", dtype="<u8", mode="r").reshape(-1, s["record_words"])
tx = rec[:, 4 + s["names"].index("tx_rdma_ucast_bytes")]
```

## Report

```
ainic-netmon-report report DIR_nodeA DIR_nodeB -o report.html
ainic-netmon-report report DIR -o zoom.html --window 12.5,12.8      # raw 2 ms points
ainic-netmon-report csv DIR --nic benic1p1 --source rdma --counters tx_rdma_ucast_bytes --rate
```

(`report/ainic_netmon_report.py`; needs `numpy` and `plotly`.)

- One panel group per node and NIC: RDMA payload rates, congestion/retransmission rates, and
  with `ipc` the wire rates and PFC pause counters. The default view covers the whole recording
  as a min/max envelope per time bin (size budget `--max-mb`); `--window` shows raw points.
- Sample time = `t0 + f * rtt` (f = 0.70 rdma, 0.25 ipc, measured). Requests slower than
  1.5x the median have no reliable snapshot time and are skipped for rates (the interval just
  becomes longer; cumulative counters lose nothing). Intervals longer than 3x the sampling
  interval are low resolution (dashed). Peak, p99 and time-above-threshold use
  normal-resolution intervals only and come with their coverage.
- At 2 ms, single-sample rates on steady traffic scatter by roughly ±5 % (rdma) and ±3 % (ipc);
  10 ms peaks are within ~3 %.
