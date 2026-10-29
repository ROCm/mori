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
metadata via C++ `RegisterMemory`; real pointers use existing classification.
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

For staged SSD load/offload, `file_cpu` / `cpu_file` describe storage-path cost;
normal CPU/GPU/network transfers add their own costs. Fit those SSD coefficients
to `SSDTier` device I/O plus its skipped payload work (record building, CRC, bounce
copies and the configured durability mode), excluding control and external
GPU/RDMA copies that the simulator already counts. Local direct file transfers use
the same coefficients for `PosixEngine`, so workloads mixing both storage paths
need separate validation. `file_gpu` models the opt-in GDS read path
(`UMBP_ENABLE_GDS=1`); GPU offload remains GPU→DRAM→SSD, matching the real engines.
The example SSD values are estimates, not calibration results.

| Field | Meaning |
| --- | --- |
| `channel` | Same names share a service queue within one backend client/server. |
| `latency_ns` | Nonnegative fixed cost per submitted batch/path. |
| `copy_latency_ns` | Nonnegative cost per coalesced copy; default 0. |
| `bandwidth_bytes_per_second` | Positive effective bandwidth. |

`cpu_cpu` coefficients may be arrays indexed by copy threads minus one, clamped to the last
entry; other paths use scalars. Existing thread selection/thresholds still apply.
Channel capacity is the maximum configured bandwidth up to the larger configured
copy/write thread limit; set those limits before creating the client.

```text
overhead = latency_ns[p] + copies * copy_latency_ns[p]
private_ns = overhead + bytes * 1e9 / bandwidth[p]
service_ns = overhead + bytes * 1e9 / channel_capacity
channel_available = max(submitted, channel_available) + service_ns
finish = max(submitted + private_ns, channel_available)
```

Shared paths sum service costs; zero bytes add no delay and missing paths abort normally.

## Cross-peer sharing

For share, keep normal Master/peer configuration, distinct node IDs and reachable
addresses/ports. Use a dedicated Master, enable sim on **every peer**, and configure
`remote_read`/`remote_write` from the initiator's perspective. Separate channel names
allow independent directions. Fit target-link **Submit+Wait**, not API totals or ping
RTT when replacing estimates with measurements. GPU/network coefficients are not calibrated
by this PR.
Local copies finish in Submit; registered remote copies finish in Wait. Unregistered
CPU callers retain RDMA bounce geometry and serial staging inside Submit. Ranged
scratch and re-cache payloads are virtual; a re-cached replica can serve other peers.
Staging shares the single-thread `cpu_cpu` coefficients with local copies. The model
does not distinguish memcpy staging from non-temporal local copies; validate staged
timings separately.
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
