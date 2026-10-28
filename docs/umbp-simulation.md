# UMBP local storage and transfer simulation

Simulation estimates the time spent **inside UMBP APIs** while running the real
allocation, placement, commit/abort, lookup, LRU, lease, watermark and local tier
transition code. It reserves virtual DRAM/HBM capacity and waits for modeled
transfers without accessing KV payloads. Simulation is off by default.

This first version supports one in-process storage owner with DRAM/HBM
PageBackends, including policies containing both. It uses wall-clock delays:
heartbeats, leases and background workers retain their existing timing.

## Enable

```python
from mori.umbp import UMBPClient, UMBPConfig

config = UMBPConfig.from_environment()  # Existing capacity/page/policy configuration.
config.sim.enabled = True
config.sim.profile_path = "/path/to/sim_profile.json"
config.sim.trace_path = "/tmp/umbp-sim-worker-0.jsonl"
client = UMBPClient(config)

# Existing put/get/batch/ranged APIs and return types stay unchanged.
client.put_from_ptr(key, ptr, nbytes)
client.get_into_ptr(key, ptr, nbytes)
```

Frameworks already using `UMBPConfig.from_environment()`, such as SGLang's UMBP
store, can set these without changing their storage call sites:

```bash
export UMBP_SIM_ENABLE=1
export UMBP_SIM_PROFILE=/path/to/sim_profile.json
export UMBP_SIM_TRACE=/tmp/umbp-sim-worker-0.jsonl
```

Keep existing capacity, page size and DRAM/HBM policy settings. The default
embedded configuration serves DRAM; the legacy `ssd.enabled` flag alone does
not select an SSD backend. An effective SSD backend is rejected in sim mode.
Use `UMBP_SSD_ENABLED=0` for a plain DRAM run. A virtual GPU caller buffer must
be registered with its GPU location/device metadata using the existing C++
`RegisterMemory` API; real framework pointers can use existing classification.
No GPU is needed to allocate a virtual HBM backend.

A smoke driver, with a small caller buffer and no model, is provided:

```bash
python examples/umbp/simulate.py --trace /tmp/umbp-sim-new-run.jsonl
```

The driver requires a new trace filename. It prints the mean and median API
latencies. The usual UMBP build dependencies (including HIP) still apply; this
is not a separate CPU-only build.

## Transfer model

[`sim_profile.json`](../examples/umbp/sim_profile.json) contains **illustrative,
uncalibrated estimates**. Replace the coefficients with measured values.
Each configured path (`cpu_cpu`, `cpu_gpu`, `gpu_cpu`, `gpu_gpu`) has:

- `latency_ns`: nonnegative fixed overhead per submitted batch on that path.
- `bandwidth_bytes_per_second`: positive effective bandwidth.
- `channel`: paths sharing this name serialize; different names may overlap.

For the total bytes on a path within one `TransferEngine::Submit`:

```text
service_ns = latency_ns + bytes * 1e9 / bandwidth_bytes_per_second
start = max(now, channel_available)
finish = start + service_ns
```

Foreground copies and background tier transitions use the same engine and
channel queues. Submit waits until all its channels finish, matching existing
synchronous local copies. Fragmenting one transfer into pages does not multiply
fixed latency. One UMBP batch API may issue multiple Submit calls; its existing
control flow determines those boundaries. Zero-byte work adds no modeled delay.
Missing paths reject transfers and the existing failure path aborts reservations.

Use a few measured sizes to fit `time = latency + size / bandwidth`; validate
against separate sizes and the relevant concurrency. Keep the profile alongside
the trace for reproducibility. Each channel currently models a serial server;
there is no device topology, dynamic bandwidth sharing or cross-process queue.

## Storage and timing records

`SimPageMemorySource` reserves inaccessible address ranges with
`mmap(PROT_NONE, MAP_NORESERVE)`. Existing PageBackend metadata represents the
keys and their pages. Configured buffer sizes are rounded down to allocatable
pages by PageBackend; hugepage/prefault settings do not allocate payload memory.
The address space reservation and page/key metadata still consume resources.

The JSONL trace records `call_id`, `api`, `key_count`, `bytes`, `start_ns`,
`elapsed_ns`, `result_count`, `exception` and `profile_path`. Times are in
nanoseconds on the monotonic clock. `bytes` is the requested API byte count
(range bytes for ranged APIs), not a claim about completed transfer volume.
`result_count` counts true results or the consecutive match count; it is `-1`
for void operations or exceptions. Existing public return values are unchanged.

Recorded operations: scalar/batch/depth/ranged put/get, exists/batch exists,
consecutive exists, register/deregister, clear, flush and close. IDs distinguish
concurrent calls within a client. Use a separate trace file per client/run;
files are appended, and IDs restart with each client. Completion order can differ
from ID order. An empty trace path writes JSON through the UMBP INFO logger.

`elapsed_ns` includes real control code, lock/queue waits and modeled transfer
waits, ending before trace serialization/output. Trace output still costs wall
time between calls. Timer/scheduler granularity can dominate very short modeled
transfers. Background work contributes when it blocks an API or competes for a
channel; its independent lifetime is not attributed to individual API calls.

## Boundaries

- Standalone-process/standalone server, SSD, RDMA and remote ranged scratch are
  rejected. Peer serving must be disabled. Remote owner connections are refused.
- A dedicated Master may register/query this single owner and receive its real
  metadata events. Use a simulation-only Master; virtual capacity must not be
  advertised to production clients. Multi-owner payload routing is unsupported.
- This version does not replace framework-owned buffers, HBM-to-host L2 copies,
  model computation, or other framework latency. It is intended for trace/dummy
  execution; restored bytes and generated tokens cannot be checked for correctness.
- The estimates depend on the control host, workload and supplied coefficients.
  Full SGLang integration and calibration are follow-up validation work, not
  accuracy guarantees established by the unit tests.

Focused checks: build `test_umbp_sim` and run `ctest -R '^umbp_sim$'` from the
build directory. Tests exercise inaccessible payload buffers, real API results,
virtual HBM, capacity/LRU/leases/reaper/events, local demotion, modeled delay and
channel contention, configuration rejection, and default real-copy behavior.
