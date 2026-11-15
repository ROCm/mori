# xgmi-netmon

Per-link XGMI bandwidth of AMD Instinct GPUs (MI355X) at 2 ms resolution: recorded locally, shown
in Grafana live or after import. Same approach as ainic-netmon (`tools/netmon`, PR #1000) for the
NICs. A standalone C++17 tool: libc only, no root, no dependency on the rest of mori or ROCm.

| Path | What |
|---|---|
| `src/` | the `xgmi-netmon` binary: `list`, `record`, `import` |
| `tests/` | unit tests (no GPU needed), `gpu_metrics` captures from MI355X |
| `peer_copy_traffic.py` | torch peer copies with known byte counts, to check the numbers |
| `../../examples/monitoring/xgmi_netmon/` | VictoriaMetrics (:28428) + Grafana (:23000) and the dashboard |

## Build

```
cmake -S tools/xgmi_netmon -B build-xgmi -DCMAKE_BUILD_TYPE=Release
cmake --build build-xgmi -j
ctest --test-dir build-xgmi          # -DXGMI_NETMON_TSAN=ON for a ThreadSanitizer build
```

`build-xgmi/xgmi-netmon` links libstdc++ statically, so it can be copied to other hosts.

## Use

```
examples/monitoring/xgmi_netmon/up.sh /tmp/xgmi-netmon-data    # local disk, not NFS
xgmi-netmon list                                     # GPUs, XGMI port -> peer GPU, link max
xgmi-netmon record -o /shared/exp42/{host} -d 60s    # record only
xgmi-netmon record -o /shared/exp42/{host} --push http://127.0.0.1:28428   # and show live
xgmi-netmon import /shared/exp42/nodeA /shared/exp42/nodeB [--replace]
examples/monitoring/xgmi_netmon/down.sh                        # keeps the data directory
```

- Run the recorder on the host, not in a container (it reads `/sys/class/drm`). No root needed.
- `{host}` expands to the host id (first DNS label of the hostname, or `--host-id`). The
  experiment defaults to the name of the parent directory (`exp42`); `--experiment` sets it.
  The output directory must not exist or be empty: a recording is never overwritten.
- `--push` refuses to start if VictoriaMetrics is unreachable or already has data for this
  experiment and host. While recording, a slow or failed push only drops points (counted in
  `meta.json`); the recording is never affected. A push gives up after 2 s without progress or
  3 s in total. A request that gave up may have been written in part, so it is never resent:
  re-import the recording with `--replace` for complete data.
- The live push sends about 2 MB/s for 8 GPUs (17 bytes per point). Run VictoriaMetrics on the
  same host or network; a push over a slow link (e.g. an ssh tunnel between sites) falls behind
  and drops points. `record` + `import` works over any link.
- `import` writes the same points as the live push (same code; checked point by point). It
  refuses an experiment and host that already have data; `--replace` deletes their `xgmi_*`
  series first. Imported data becomes visible in VictoriaMetrics after up to ~10 s.
- View: `ssh -L 23000:127.0.0.1:23000 <grafana host>`, open `http://localhost:23000`, pick
  experiment, host and the time range.

## Output

```
DIR/.lock          created exclusively: pid, host, start time
DIR/meta.json      status (running | complete | failed), host, experiment, GPUs and peers,
                   per-GPU statistics, push statistics, CPU time
DIR/gpu<N>.bin     512-byte header (magic XGMINM01, GPU, peers, link max, host, experiment),
                   then records of 21 little-endian u64:
                   t0_mono_ns, t1_mono_ns, real_ns, real_err_ns, firmware_timestamp (10 ns),
                   xgmi_read_data_acc[8], xgmi_write_data_acc[8] (KB)
```

A record is written only when `firmware_timestamp` changed. A recording killed mid-write may end
with a partial record; `import` drops it. A recorder that died leaves `status: running`.

## How it samples

- One thread per GPU wakes on the wall-clock grid (multiples of 2 ms in `CLOCK_REALTIME`, like
  ainic-netmon) with `clock_nanosleep(TIMER_ABSTIME)` and `pread`s `gpu_metrics`. It never blocks
  and never allocates: it hands records to a writer thread and a push thread through lock-free
  rings; a full ring drops the record and counts it.
- `gpu_metrics` v1.8 (fixed struct, amdgpu 6.16.6) and v1.9 (self-describing, amdgpu 6.16.13;
  offsets parsed at startup) are supported.

## What the numbers are

- PMFW refreshes `gpu_metrics` every 2 ms. Rate = delta bytes / delta `firmware_timestamp`, not
  host time. A missed snapshot loses no bytes; the interval just gets longer.
- No point for an interval where a counter went down (driver reload) or where the rate is above
  2x the link max (not physical); both are counted.
- `xgmi_write_data_acc` is what a GPU sent on a link, `xgmi_read_data_acc` what it received. Both
  count link bytes, not payload.
- Use the sender's `xgmi_write`. Checked with torch peer copies of known size on n06-21, GPUs 4-7,
  33 isolated random segments of 1-2 s (1-4 pairs at once, 16/64/256 MB blocks):

  | | counter / copied bytes |
  |---|---|
  | one-way copy, sender's `xgmi_write` | 1.1343 ± 0.0003, same for every block size |
  | reverse link of a one-way copy | 0.569 ± 0.001 |
  | both directions copying at once | 1.14–1.43: no fixed factor |
  | receiver's `xgmi_read` / sender's `xgmi_write`, same link | 0.70–1.30 per segment |

  The receiver's `xgmi_read` follows the sender within 0-10 ms when one flow arrives, but when a
  GPU receives several flows at once it can be off by up to 30 % in a 1-2 s window (per link and in
  total; over 60 s each link stays within 1.6 %). Cause unknown. So "received" is computed as what
  the peers' `xgmi_write` sent to the GPU.
- Payload = sent / 1.134 holds only for one-way traffic (1.118 on Crusoe v3-088, a VM with a
  newer driver). One copy ran at 61 GB/s payload; the link max is 72 GB/s per direction.
- Series: `xgmi_bytes`, `xgmi_rate_bytes_per_second` (labels experiment, host, gpu, peer,
  dir = wr | rd) and `xgmi_interval_seconds` (per GPU, with the labels `links` and
  `link_max_bytes_per_second`: the link capacity from `gpu_metrics`, for the dashboard's reference
  lines). Point time = wall clock of the read
  that first saw the snapshot. Both the push and `import` write VictoriaMetrics' `/api/v1/import`
  format, one JSON line per series (as ainic-netmon), so labels are sent once per request.

## Dashboard

Pick experiment, host, GPU and two variables:

- **resolution**: the finest bin of the time charts (auto = 6 ms, 10 ms, 100 ms, 1 s, 5 s; the data
  itself is 2 ms, coarser bins average more).
- **average window**: what the tiles, tables, matrix and bars average over: the selected range, or
  only the last 1 s / 10 s / 1 min before its end (for live use, where the range is minutes long).

Top to bottom:

- Tiles: total sent, total load (share of the capacity), busiest GPU, hottest link and data age.
  *Busiest GPU* is the GPU with the highest average sent bandwidth (its 7 links added) in the
  window. *Hottest link* is the link with the highest single 2 ms interval in the window, so it is
  a peak, not an average. *Data age* is the time from the newest sample (within 1 h) to the end of
  the range: in live mode 1-6 s is normal (VictoriaMetrics makes new data searchable after about
  5 s), red from 30 s means the recorder or the push stalled.
- GPU overview table (like rdmatop's device table): per GPU the average sent and received
  bandwidth, each also as a bar of the GPU's capacity (links x link max), the highest single 2 ms
  interval on any link, the coverage (the share of the window with valid sampling intervals; below
  100 % the numbers cover only part of the window) and a trend of the sent bandwidth. Bar colours as
  in rdmatop: under 5 % grey, 5-40 % dim green, 40-70 % green, 70-85 % yellow, from 85 % red.
- Sent matrix: row GPU -> column GPU (colour scale 0-72 GB/s).
- Sent and received per GPU as bars (average GB/s).
- Sent GPU -> GPU over time, and sent per GPU.
- One row per GPU: the sent bandwidth to each peer over time with the 2 ms peak (red dashed), and
  a link table (one row per link: sent, share of one link's capacity, 2 ms peak, trend).
- Diagnostics: the longest sampling interval per GPU.

All panels use the senders' `xgmi_write`; received is what the peers sent to the GPU. The time
charts are steps: a point is the average of the bin that ends at it, so the value is drawn over
that bin, not as a slope between points. The grey dashed lines are the capacity of one link and of
all links of a GPU, read from the `links` / `link_max_bytes_per_second` labels (data recorded
before they existed has no reference lines, no load bars and no total load). The average panels
use exact milliseconds (`${__range_ms}ms`; Grafana's `$__range` is whole seconds and would cut the
start).

## Measured on n06-21 (MI355X, 8 GPUs, idle)

| | |
|---|---|
| snapshots per GPU | 489–497 per s (500 is every PMFW snapshot) |
| intervals > 3 ms | 0.02 %–1.1 % per 6–30 s run (0.0–0.1 % on Crusoe v3-088, v1.9) |
| read start after the grid point | ~55 µs on every GPU |
| CPU | 0.46–0.64 core kernel time (the sysfs reads), 0.05 core user; 0.12–0.18 core user with `--push` |
| live push | 1.97 MB/s, 17.4 bytes per point |
| live push vs import | 232 series, 3 435 398 points, identical |

## Known limits

- KB = 1024 bytes is assumed; it cannot be separated from the 1.12x link overhead with this test.
- The effect of reading `gpu_metrics` every 2 ms on a running workload is not measured yet.
- Point times are read times: up to ~2 ms after the snapshot.
- Deployment uses `docker run`, not compose (the `docker-compose` 1.29 on n06-21 is broken).
  Both services listen on 127.0.0.1 only; Grafana allows anonymous admin, do not expose it.
  `up.sh` copies the Grafana files into the data directory, because the docker daemon cannot read
  an NFS home with root squash; copy `xgmi.json` there again after editing it.
