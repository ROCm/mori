# UMBP storage and transfer simulation

Simulation keeps allocation, placement, commit/abort, lookup, LRU, leases, events,
tier policies and control RPCs. Payload storage/transfers are virtual; sim is off by default.

## Enable

Keep existing capacities, policies and framework buffers; add the sim settings:

```python
from mori.umbp import UMBPClient, UMBPConfig

config = UMBPConfig.from_environment()
config.sim.enabled = True
config.sim.profile_path = "/path/to/profile.json"
config.sim.trace_path = "/tmp/umbp-worker-0.jsonl"
client = UMBPClient(config)
client.put_from_ptr(key, ptr, nbytes)
client.get_into_ptr(key, ptr, nbytes)
```

With `UMBPConfig.from_environment()`, the equivalent settings are `UMBP_SIM_ENABLE=1`,
`UMBP_SIM_PROFILE=/path/to/profile.json` and `UMBP_SIM_TRACE=/tmp/umbp-worker-0.jsonl`.
Existing APIs and return types remain available. Use a separate trace file per client/run.
DRAM, HBM and file-backed SSD tiers are supported. Embedded mode defaults to DRAM;
select SSD through the existing backend policy or distributed medium configuration;
`ssd.enabled` alone does not select SSD. Virtual GPU callers require location/device
metadata via C++ `RegisterMemory`; unregistered real pointers use existing classification.
Virtual HBM needs no GPU, but normal build dependencies, including HIP, still apply.

SSD uses the existing `SsdBackend`, `PeerSsdManager`, segment index and multi-drive
placement. Its files and staging payloads are virtual: `storage_dir` identifies
shards but no directories/files are created. Capacity includes record headers,
4 KiB padding and dead records; deleting a key frees disk capacity when its whole
segment becomes reclaimable. Commit/abort, LRU, staging limits, read leases and
single-flight reads keep their normal behavior. Simulation starts with an empty
tier and does not restore persisted data. SPDK backends are outside this file-tier
implementation.

### Standalone / CI

Run one server per node; its capacity, page size, LRU and policies apply to all ranks:

```bash
export UMBP_SIM_ENABLE=1 UMBP_SIM_PROFILE=/path/to/profile.json
export UMBP_SIM_TRACE=/tmp/umbp-server.jsonl
umbp_standalone_server unix:/tmp/umbp.sock
```

Ranks keep their `standalone_process` config and socket, enable `config.sim.enabled`,
and use distinct trace files. Forwarding ranks need no profile. For `auto_start=True`,
set the profile in the client; the server receives it and uses trace suffix `.server`.
Both sides need the updated build and matching sim mode. Range checks remain active;
closing a rank releases registrations while other ranks retain the shared KV tier.
Rank timings include IPC and server work; **do not add server trace times to rank times**.

## Profile and model

[sim_profile.json](../examples/umbp/sim_profile.json) contains illustrative, uncalibrated
estimates for `cpu_cpu`, `cpu_gpu`, `gpu_cpu`, `gpu_gpu`, `remote_read`, `remote_write`,
`file_cpu`, `cpu_file` and `file_gpu`.
Copy it and adjust the coefficients for your platform.
Each path names its source/destination memory type, except the remote paths, which use
the initiator's read/write direction. Configure every path your workload uses; a missing
path rejects the transfer. Unused paths may be omitted.

This first version uses manually supplied parameters. Automatic calibration belongs in
a follow-up PR covering the actual engines (`LocalCopyEngine`, `HbmCopyEngine` and
`MoriIoEngine`), with reusable profiles for CPU, GPU and RDMA transfers. SSD coefficients
also need measurements of the actual configured storage path.

For staged SSD load/offload, optional `file_cpu_staged` / `cpu_file_staged` describe storage-path cost;
normal CPU/GPU/network transfers add their own costs. Fit those SSD coefficients
to `SSDTier` device I/O plus its skipped payload work (record building, CRC, bounce
copies and the configured durability mode), excluding control and external
GPU/RDMA copies that the simulator already counts. Local direct file transfers use
`file_cpu` / `cpu_file`; profiles without staged variants fall back to those curves.
`file_gpu` models the opt-in GDS read path
(`UMBP_ENABLE_GDS=1`); GPU offload remains GPU→DRAM→SSD, matching the real engines.
Checksum-bearing whole reads use `file_cpu_staged`. Checksum-free records and
partial reads prefer `file_cpu_staged_no_crc`, which models the same SSDTier I/O
backend without CRC work. Older profiles fall back to `file_cpu`, then the staged
curve for these reads; whole reads fall back to `file_cpu`.
With CRC disabled, calibrate the staged profile with the same setting.
The example SSD values are estimates, not calibration results.

| Field | Meaning |
| --- | --- |
| `channel` | Same names share a service queue within one backend client/server. |
| `latency_ns` | Nonnegative fixed cost per submitted batch/path. |
| `copy_latency_ns` | Nonnegative cost per coalesced copy; default 0. |
| `registration_latency_ns` | Optional nonnegative first-FD cost for `file_gpu`; default 0. |
| `log_size_latency_ns` | Optional nonnegative SSD read cost per copy × `log2(max(1, mean copy bytes / 4096))`; default 0. |
| `batch_log_size_latency_ns` | Optional nonnegative SSD read cost per batch × the same size term; default 0. |
| `serialize_latency` | Optional SSD read flag to queue latency as well as bytes, for mutually exclusive I/O batches; default false. |
| `bandwidth_bytes_per_second` | Positive effective bandwidth. |

`cpu_cpu` coefficients may be arrays indexed by copy threads minus one, clamped to the last
entry; other paths use scalars. Existing thread selection/thresholds still apply.
Channel capacity is the maximum measured bandwidth; independent callers can use
more cores than one call's copy/write thread limit. Each call retains its thread curve.

```text
size_term = log2(max(1, bytes / copies / 4096))
overhead = latency_ns[p] + copies * copy_latency_ns[p]
         + size_term * (copies * log_size_latency_ns + batch_log_size_latency_ns)
private_ns = overhead + bytes * 1e9 / bandwidth[p]
service_ns = overhead + bytes * 1e9 / channel_capacity
channel_available = max(submitted, channel_available) + service_ns
finish = max(submitted + private_ns, channel_available)
```

Shared paths sum service costs; zero bytes add no delay and missing paths abort normally.
For independent SSD reads, `service_ns` includes only the byte cost so request
latency can overlap, unless `serialize_latency` is true. Writes and mixed read/write
submits retain the full service cost.

Pure `file_gpu` submits with `serialize_latency=false` share channel capacity among
active synchronous copies. Copy/log latency stays private; fixed batch/log latency
is paid once, using the same five coefficients. Ordinary reservations on that
channel form a barrier; mixed submits retain the formulas above. File registration
timing is unchanged.

Native CRC reads with a matching serialized no-CRC curve on the same channel
queue the smaller of the two costs as I/O, then finish the remaining CRC cost
outside that queue. Missing or incompatible curves retain full serialization.

`cpu_file_staged` uses the complete payload curve: record preparation, I/O,
durability and record-buffer cleanup are included in its measured total cost.
The basic model reserves that combined cost on the SSD channel; it does not model
preparation or cleanup overlap between callers separately.

## Cross-peer sharing

For share, keep normal Master/peer configuration, distinct node IDs and reachable
addresses/ports. Use a dedicated Master, enable sim on **every peer**, and configure
`remote_read`/`remote_write` from the initiator's perspective. Separate channel names
allow independent directions. Fit target-link **Submit+Wait**, not API totals or ping
RTT when replacing estimates with measurements. The example GPU/network coefficients
remain uncalibrated; use the separate calibration workflow for platform measurements.
Local copies finish in Submit; registered remote copies finish in Wait. Unregistered
CPU callers retain RDMA bounce geometry and serial staging inside Submit. Ranged
scratch and re-cache payloads are virtual; a re-cached replica can serve other peers.
Staging uses optional scalar `cpu_staging` / `cpu_staging_read` coefficients calibrated
from MoriIo's serial copyin/copyout geometry. Reads fall back to `cpu_staging`, then
the single-thread `cpu_cpu` estimate. Validate staged timings separately from local copies.
Sim/real engines reject each other's handshake. Distributed ranged I/O still needs scratch.
Remote GPU callers retain real registration restrictions; standalone GPU IPC is local-copy-only.

## Timing and limits

JSONL fields are `api`, `key_count`, requested `bytes` (range bytes for ranged calls),
monotonic `start_ns`/`elapsed_ns`, `result_count` and `exception`. Results count successes/
matches; void calls and exceptions use -1. Records cover scalar/batch/depth/ranged
put/get, existence, register/deregister, clear, flush and close. Completion order may
differ. Files append and flush on flush/close/destruction; an empty path uses INFO
logging, and disabling INFO skips formatting.

`elapsed_ns` includes real control code, RPC/locks, queueing and modeled transfers,
ending before trace serialization; whole-API timing also includes trace output.
Workers/timers use wall time; background work counts when it delays an API/channel.

- Queues belong to one backend client/server, shared by its ranks. Remote-NIC aggregation, switch congestion and dynamic fair sharing are not modeled.
- Mixed transfer paths on different channels may overlap in the model; real inline engines can serialize them. Validate mixed-tier batches separately.
- Real control RPC failures remain possible; modeled payload-transfer failures are not synthesized. Same-host peers measure that host's RPC cost.
- Scheduling/control overhead limits short-operation accuracy. Profiles depend on platform, concurrency and workload.
- Virtual addresses and page/key metadata consume resources. Framework buffers, external copies and model execution remain real; use trace/dummy execution, not generated-token correctness checks.
- GPU/network accuracy and full SGLang performance validation require separate work.

Build `test_umbp_sim` and run `ctest -R '^umbp_sim$'`; these functional tests need no GPUs.
