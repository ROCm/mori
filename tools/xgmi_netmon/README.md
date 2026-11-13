# xgmi-netmon

Per-link XGMI bandwidth of AMD Instinct GPUs (MI355X) at 2 ms resolution, shown in Grafana.
Same idea as ainic-netmon (`tools/netmon`, PR #1000) for the NICs: VictoriaMetrics + Grafana, live push.

| File | What |
|---|---|
| `xgmi_netmon.py` | collector: `list` (topology), `push` (sample and push to VictoriaMetrics) |
| `examples/monitoring/xgmi_netmon/up.sh DATA_DIR`, `down.sh` | start / remove VictoriaMetrics (:28428) + Grafana (:23000) |
| `examples/monitoring/xgmi_netmon/grafana/` | datasource, dashboard provisioning, dashboard `xgmi.json` |
| `peer_copy_traffic.py` | torch peer copies with known byte counts, to check the numbers |

## Use

```
examples/monitoring/xgmi_netmon/up.sh /tmp/xgmi-netmon-data   # local disk: VictoriaMetrics does not work on NFS
python3 tools/xgmi_netmon/xgmi_netmon.py list    # GPUs, XGMI port -> peer, link max
python3 tools/xgmi_netmon/xgmi_netmon.py push    # until Ctrl-C; -d 60 for 60 s; --host NAME
examples/monitoring/xgmi_netmon/down.sh                         # removes the containers, keeps DATA_DIR
```

- No root needed (`gpu_metrics` is world-readable). Run the collector on the host, not in a container.
- Another node: open a reverse tunnel from the Grafana host, `ssh -N -R 28428:127.0.0.1:28428 NODE`,
  then run `push` on NODE (default URL `http://127.0.0.1:28428`).
- View: `ssh -L 23000:127.0.0.1:23000 <grafana host>`, open `http://localhost:23000`. Pick `host`
  and the time range; the default is the last 5 minutes.
- Both services listen on 127.0.0.1 only; Grafana allows anonymous admin, do not expose it.
- `up.sh` copies its `grafana/` directory into `DATA_DIR/conf` (the docker daemon cannot read an NFS home
  with root squash). After editing the dashboard, copy `xgmi.json` there again; Grafana reloads it.
- It uses `docker run`, not compose: the `docker-compose` 1.29 on n06-21 fails with
  "Not supported URL scheme http+docker".

## Dashboard

- Sent matrix: row GPU -> column GPU, average over the selected range.
- Sent GPU -> GPU over time: one line per link.
- Sent per GPU: all links of one GPU.
- One row per GPU: sent to each peer, plus the highest single 2 ms interval (red dashed).
- Diagnostics (collapsed): longest sampling interval per GPU.

Only sent is shown: the receiver's `xgmi_read` equals the sender's `xgmi_write` (checked on both
ends of every link). The collector still pushes both.

## What the numbers are

- Source: `gpu_metrics` v1.8 (fixed struct, amdgpu 6.16.6) or v1.9 (self-describing, amdgpu
  6.16.13; offsets parsed at startup). Fields: `xgmi_write_data_acc` / `xgmi_read_data_acc`
  (KB, cumulative), `firmware_timestamp` (10 ns).
- PMFW refreshes the table every 2 ms. Rate = delta bytes / delta `firmware_timestamp`, not host
  time. A missed snapshot loses no bytes; the interval gets longer.
- Values are link bytes, not payload. With `peer_copy_traffic.py`: a copy shows 1.118x its payload
  in its direction and 0.569x in the opposite direction. One GPU0->GPU1 copy ran at 59.9 GB/s payload
  (about 67 GB/s on the link; the link max is 72 GB/s per direction).

## Known limits

- Sampling is every 2 ms on the wall-clock grid, as ainic-netmon. Host and PMFW clocks are not
  phase-locked, so how many snapshots are missed depends on the phase between the two 2 ms grids:
  on n06-21, 5 % to 63 % of the intervals were longer than 3 ms in 2 s runs (18 % over 30 s).
  No bytes are lost; the affected intervals are 4 ms instead of 2 ms. `--period-ms 1` gave
  9 % > 3 ms over 30 s at higher CPU cost.
- CPU: 8 GPUs at 2 ms cost about 0.6 core of kernel time (0.85 core at 1 ms).
- One round reads the 8 GPUs one after another (about 1.2 ms), and the snapshot is up to 2 ms
  older than the read. Point times are the read time.
- KB = 1024 bytes is assumed. It cannot be separated from the 1.12x protocol overhead with this test.
