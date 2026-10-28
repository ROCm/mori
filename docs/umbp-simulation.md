# UMBP storage and transfer simulation

Simulation estimates the time spent **inside UMBP APIs** while running the real
allocation, placement, commit/abort, lookup, LRU, lease, watermark and local tier
transition code. It reserves virtual DRAM/HBM capacity and waits for modeled
transfers without accessing KV payloads. Simulation is off by default.

This version supports in-process storage owners with DRAM/HBM PageBackends,
including policies containing both and sharing KV across peers. It uses wall-clock delays:
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

## Calibrate on the target platform

The optional calibration tool implements two stages: real sampling and profile
fitting, followed by simulation with those saved parameters. It currently
calibrates **CPU→CPU copies only**. It does not change runtime APIs or perform
calibration during client initialization.

Build the sampler in an existing Release build with UMBP enabled, then run the
Python driver (requires NumPy and Linux `taskset`):

```bash
cmake --build build --target umbp_sim_probe -j 8
python examples/umbp/calibrate.py \
  --probe build/src/umbp/umbp_sim_probe \
  --output /tmp/umbp-calibration \
  --cpus 4,5,6,7 --threads 1 2 4 \
  --cache cold --samples 24 --rounds 3
```

Use a new output directory and available physical CPU cores representative of
deployment. Each N-thread probe runs on the first N listed CPUs and sets both
`UMBP_DRAM_COPY_THREADS` and `UMBP_DRAM_WRITE_THREADS` to N. Thread counts must
include 1; supported sampling points are 1, 2, 4 and 8. Without explicit options,
the tool uses up to eight allowed CPUs and those thread counts that fit. Preserve
the same copy settings, CPU/NUMA placement and workload conditions for simulation.

The tool writes:

- `profile.json`: load directly using `UMBP_SIM_PROFILE`. It contains the fitted
  fixed latency, per-copy cost and bandwidth for `cpu_cpu`, with interpolation
  between measured thread counts. Other transfer paths remain unconfigured.
- `report.json`: platform, CPU affinity/topology, copy environment, profile/probe
  hashes, raw samples and independent real/sim comparisons. Each comparison
  includes medians, p95s and signed median error; the reported tolerance is
  `max(2 µs, 10% of the real median)`. High error remains visible and does not
  prevent writing the profile; failed operations or invalid fits fail the run.

Training uses real `LocalCopyEngine` Submit+Wait timings across 4 KiB–4 MiB
blocks and batches of 1, 8 and 32, capped at 64 MiB total. Only these measurements
fit the nonnegative model. Separate 8 KiB–8 MiB blocks and batches of 1, 4 and 16
validate transfers and actual `BatchPut`/`BatchGet` calls. Existing allocation,
lookup and other control costs still run; API totals are not added to the model.
API validation includes simulation trace formatting and buffered `/dev/null`
output, but excludes Python/framework overhead.

`cold` flushes the caller's source outside the timer before each transfer/put
(x86 only). Get reads the just-written store; it is not an independently flushed
read. `reuse` repeats with the same buffers and limits training to 8 MiB total;
larger validation working sets are flagged in the report. Reuse does not
guarantee a cache hit. Neither profile infers changing cache state or system load.
In particular, a reused-source engine profile can underestimate store reads:
the engine's source may stay cached while data just written to the store does
not. Use the API error report to judge suitability, not just the transfer fit.
GPU copies and framework end-to-end accuracy require separate calibration.

The simulation stage only loads the generated profile:

```bash
export UMBP_SIM_ENABLE=1
export UMBP_SIM_PROFILE=/tmp/umbp-calibration/profile.json
export UMBP_SIM_TRACE=/tmp/umbp-sim-worker-0.jsonl
# Keep existing UMBP configuration and put/get call sites.
```

## Transfer model

[`sim_profile.json`](../examples/umbp/sim_profile.json) contains **illustrative,
uncalibrated estimates**. Replace the coefficients with measured values.
Each configured path (`cpu_cpu`, `cpu_gpu`, `gpu_cpu`, `gpu_gpu`, `remote_read`,
`remote_write`) has:

- `latency_ns`: nonnegative fixed overhead per submitted batch on that path.
- `bandwidth_bytes_per_second`: positive effective bandwidth.
- `copy_latency_ns`: optional nonnegative overhead per coalesced copy (default 0).
- `channel`: paths sharing this name share a service queue; different names may overlap.

For `cpu_cpu`, each coefficient can also be an array indexed by the number of
copy threads minus one. For example, `[4e10, 6e10, 7e10, 8e10]` supplies bandwidths
for 1, 2, 3 and 4 threads. These are illustrative values, not measurements.
Scalars retain their original meaning; arrays clamp to their last entry beyond
their calibrated range. Other paths currently use one coefficient per field.

The simulator reuses LocalCopyEngine's thread selection, including the byte
threshold, copy count and background-thread restriction. Channel capacity is
the maximum calibrated bandwidth up to the larger of `UMBP_DRAM_COPY_THREADS`
and `UMBP_DRAM_WRITE_THREADS`. Set these before creating the client, as in real
mode. A CPU path with scalar coefficients still acts as a serial channel.

For the bytes and coalesced copies on a path within one `TransferEngine::Submit`:

```text
overhead = latency_ns[p] + copies * copy_latency_ns[p]
private_ns = overhead + bytes * 1e9 / bandwidth[p]
service_ns = overhead + bytes * 1e9 / channel_capacity
channel_available = max(now, channel_available) + service_ns
finish = max(now + private_ns, channel_available)
```

`now` is the Submit entry timestamp: computing the model consumes its time
budget instead of adding a second setup cost to the calibrated duration.

This preserves a single stream's latency while allowing independent submissions,
including BatchPut's cross-key copies, to overlap within the aggregate bandwidth
limit. It approximates throughput and per-submission latency, not dynamic fair
sharing or exact completion order among competing streams.

Foreground copies and background tier transitions use the same engine and
channel queues. Local copies finish inside Submit; registered remote transfers
return a handle and finish in Wait, preserving the existing overlap with other
peers and local work. Unregistered CPU buffers use the RDMA planner's bounce
geometry and complete staging inside Submit, including modeled CPU copies.
Fragmenting one transfer into pages does not multiply
fixed submission latency. The optional per-copy cost applies after adjacent
segments have been coalesced by the real planner. One UMBP batch API may issue multiple Submit calls; its existing
control flow determines those boundaries. Zero-byte work adds no modeled delay.
Missing paths reject transfers and the existing failure path aborts reservations.

Use measured sizes, copy counts and thread settings to fit these coefficients;
validate against separate sizes and batch shapes. Keep the profile alongside
the trace for reproducibility. Calibration must also match source-buffer reuse
and working-set size: cold DRAM and a repeatedly reused cached source can have
very different effective bandwidths. There is no automatic cache-state inference,
device topology or cross-process queue.

## Cross-node sharing

Run the normal Master and configure each peer as usual: distinct node IDs,
reachable node addresses and peer-service ports, DRAM/HBM capacity, and optionally
`ranged_scratch_size`. Enable simulation and supply a profile on **every peer**.
The Master and peer RPCs still run over the real network: registration, routing,
allocation, resolve, commit/abort, leases, heartbeats and eviction are unchanged.
The configured RDMA engine is replaced with the simulator; no NIC registration
or payload traffic occurs. Simulated and real transfer engines reject each
other's handshake. Use a dedicated simulation Master to keep their metadata apart.

Add network parameters to the profile's existing `paths`. These example numbers
are **illustrative, not measured**:

```json
"remote_read": {
  "channel": "nic",
  "latency_ns": 8000,
  "copy_latency_ns": 0,
  "bandwidth_bytes_per_second": 25000000000
},
"remote_write": {
  "channel": "nic",
  "latency_ns": 8000,
  "copy_latency_ns": 0,
  "bandwidth_bytes_per_second": 25000000000
}
```

Read/write are from the initiating client's perspective. The same channel shares
capacity across peers and directions in that client; separate names model
independent directions. Network coefficients are scalars and apply to the endpoint
types represented by the measurements. Use target-link **transfer Submit+Wait**
measurements for fitting; API or ping RTT includes different control work and
must not be charged again. The packaged automatic sampler currently calibrates
CPU copies only; it does not measure RDMA or supply calibrated network values.

Ordinary and ranged load/offload use the real peer control paths. Storage pools,
ranged scratch, bounce buffers and asynchronous re-cache payloads are virtual.
Remote reads can still create local replicas using the existing admission and
queue policies, and those replicas can subsequently serve other peers. The
same JSONL API trace includes real RPC/control time, modeled transfer waits and
any on-path staging. Waits that overlap are not summed twice.

Channel queues are local to each client. This models one initiator's configured
link capacity; it does not aggregate different initiators contending for the same
remote NIC or model switch congestion. Transport failures during a modeled delay
are not synthesized; actual control RPC failures still follow normal failure paths.
Running peers on one machine exercises the control protocol but measures that
machine's RPC costs, not the RPC latency of a different physical cluster.

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
File output is buffered; `flush`, `close`, and client destruction flush remaining
records. Disabling INFO with no trace file skips record formatting. Paths are
JSON-escaped once at initialization, and per-call integer fields use lightweight
formatting without a protobuf object allocation.

`elapsed_ns` includes real control code, lock/queue waits and modeled transfer
waits, ending before trace serialization/output. Trace output still delays the
API return, so a caller timing the whole call also pays that overhead.
The transfer wait sleeps until 100 microseconds before its
completion deadline, then busy-waits for the remaining tail. This avoids the
roughly 50-microsecond sleep overshoot seen on the initial CPU test host, at the
cost of CPU time; scheduling contention can still delay completion. Very short
operations can also be dominated by control and trace overhead. Background work
contributes when it blocks an API or competes for a channel; its independent lifetime is not attributed to individual API calls.

## Boundaries

- Standalone-process/standalone server and SSD remain unsupported. In-process
  peers use simulated payload transfers and real Master/peer control RPCs.
- This version does not replace framework-owned buffers, HBM-to-host L2 copies,
  model computation, or other framework latency. It is intended for trace/dummy
  execution; restored bytes and generated tokens cannot be checked for correctness.
- The estimates depend on the control host, workload, copy-thread count and supplied
  coefficients. A profile calibrated with one concurrency configuration does not
  establish accuracy for another.
  Full SGLang integration and GPU/network calibration are follow-up validation work, not
  accuracy guarantees established by the unit tests.

Focused checks: build `test_umbp_sim` and run `ctest -R '^umbp_sim$'` from the
build directory. Tests exercise inaccessible payload buffers, real API results,
virtual HBM, capacity/LRU/leases/reaper/events, local demotion, modeled delay and
channel contention, remote/ranged share, re-cache, eviction visibility, failed-write
rollback, distinct peer descriptors, staging, asynchronous completion, mixed-mode
rejection, configuration rejection, and default real-copy behavior.
