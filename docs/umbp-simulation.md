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
Only DRAM/HBM backends are supported; SSD is rejected. Embedded mode defaults to DRAM;
`ssd.enabled` alone does not select SSD. Virtual GPU callers require location/device
metadata via C++ `RegisterMemory`; real pointers use existing classification.
Virtual HBM needs no GPU, but normal build dependencies, including HIP, still apply.

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

## Calibrate once, reuse the profile

The packaged sampler fits **CPU→CPU only**. Use a Release build, NumPy, Linux `taskset`,
a new output directory and available physical CPUs representative of deployment:

```bash
cmake --build build --target umbp_sim_probe -j 8
python examples/umbp/calibrate.py \
  --probe build/src/umbp/umbp_sim_probe --output /tmp/umbp-calibration \
  --cpus 4,5,6,7 --threads 1 2 4 --cache cold \
  --samples 24 --rounds 3 --max-mib 256 --concurrency 1 2 4
```

| Option | Meaning and defaults |
| --- | --- |
| `--cpus`, `--threads` | Threads are 1/2/4/8, including 1; defaults use up to eight allowed CPUs. Each N-thread run sets both `UMBP_DRAM_COPY_THREADS` and `UMBP_DRAM_WRITE_THREADS` to N. |
| `--concurrency` | 1/2/4 callers, default all three, including 1. Use `1` to skip concurrent validation. |
| `--max-mib` | Aggregate payload across all callers, default 256 MiB; destinations and storage pools need additional memory. |
| `--samples`, `--rounds` | Samples per workload and repeated rounds; defaults 24 and 3. |
| `--train-block-bytes`, `--train-batches` | Positive integer training lists, e.g. `4096 65536 1048576` and `1 8 32`. |
| `--validation-block-bytes`, `--validation-batches` | Independent validation lists, e.g. `8192 131072 2097152` and `1 4 16`. |
| `--cache` | `cold` (default, x86 source flush outside timers) or `reuse`. |

| Default workload | Block sizes | Batch counts | Uncapped grid maximum |
| --- | --- | --- | --- |
| Training | 512 B–4 MiB | 1/8/32/64/128 | 512 MiB |
| Single-caller validation | Independent sizes, 1 KiB–8 MiB | 1/4/16/128 | 1 GiB |
| Concurrent validation | 8 KiB, 512 KiB, 2 MiB | 1/16/128 | Subject to the aggregate cap |

Custom lists sample their full block/batch product, including concurrent validation;
omitted lists retain defaults and duplicates are sampled once. Training and validation
must not share a sampled `(block_bytes, batch)` pair. Overlap, empty capped grids or
insufficient independent training points fail without publishing a profile. Increase the
cap/lists to cover the working set; copy blocks do not represent full layer/ranged geometry.

Callers share one client/engine with disjoint keys/buffers and start together; thread
creation is outside timers. Runs use the first `min(N * callers, listed CPUs)` cores,
which can oversubscribe a short CPU list. Reads and writes are tested separately;
independent ranks/processes and simultaneous read/write mixes are outside this sweep.
Keep copy settings and CPU/NUMA placement consistent when reusing a profile.

`cold` flushes sources before transfer/put; get reads the just-written store. `reuse`
repeats buffers and caps default training at 8 MiB; explicit lists use `--max-mib`.
Neither mode guarantees cache hits; a reused-source fit may underestimate store reads,
so inspect API validation too.

Only real single-caller `LocalCopyEngine` Submit+Wait training fits coefficients.
Independent validation compares transfers and `BatchPut`/`BatchGet`; API timings include
trace output to `/dev/null` and exclude Python. `profile.json` loads via `UMBP_SIM_PROFILE`.
`report.json` records sampling parameters, profile hash, raw measurements and
per-concurrency median/signed errors. Tolerance is `max(2 µs, 10% of real median)`;
high error is reported, while operation/fit failures fail the run.

## Model and cross-peer sharing

[sim_profile.json](../examples/umbp/sim_profile.json) contains illustrative, uncalibrated
values. Paths are `cpu_cpu`, `cpu_gpu`, `gpu_cpu`, `gpu_gpu`, `remote_read`, `remote_write`:

| Field | Meaning |
| --- | --- |
| `channel` | Same names share a service queue within one backend client/server. |
| `latency_ns` | Nonnegative fixed cost per submitted batch/path. |
| `copy_latency_ns` | Nonnegative cost per coalesced copy; default 0. |
| `bandwidth_bytes_per_second` | Positive effective bandwidth. |

CPU coefficients may be arrays indexed by copy threads minus one, clamped to the last
entry; other paths use scalars. Existing thread selection/thresholds still apply.
Channel capacity is the maximum calibrated bandwidth up to the larger configured
copy/write thread limit; set those limits before creating the client.

```text
overhead = latency_ns[p] + copies * copy_latency_ns[p]
private_ns = overhead + bytes * 1e9 / bandwidth[p]
service_ns = overhead + bytes * 1e9 / channel_capacity
channel_available = max(submitted, channel_available) + service_ns
finish = max(submitted + private_ns, channel_available)
```

Shared paths sum service costs; zero bytes add no delay and missing paths abort normally.

For share, keep normal Master/peer configuration, distinct node IDs and reachable
addresses/ports. Use a dedicated Master, enable sim on **every peer**, and configure
`remote_read`/`remote_write` from the initiator's perspective. Separate channel names
allow independent directions. Fit target-link **Submit+Wait**, not API totals or ping
RTT; GPU/network coefficients require separate measurements, with no automatic RDMA sampler.
Local copies finish in Submit; registered remote copies finish in Wait. Unregistered
CPU callers retain RDMA bounce geometry and serial staging inside Submit. Ranged
scratch and re-cache payloads are virtual; a re-cached replica can serve other peers.
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
- Real control RPC failures remain possible; modeled payload-transfer failures are not synthesized. Same-host peers measure that host's RPC cost.
- Scheduling/control overhead limits short-operation accuracy. Profiles depend on platform, concurrency and workload.
- Virtual addresses and page/key metadata consume resources. Framework buffers, external copies and model execution remain real; use trace/dummy execution, not generated-token correctness checks.
- GPU/network accuracy and full SGLang performance validation require separate work.

Build `test_umbp_sim` and run `ctest -R '^umbp_sim$'`. Calibration math tests are in
`tests/python/umbp/test_sim_calibration.py`; neither functional suite needs GPUs.
