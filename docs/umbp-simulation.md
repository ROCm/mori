# UMBP storage and transfer simulation

Simulation runs the real allocation, placement, commit/abort, lookup, LRU,
leases, events and tier policies. Only payload storage and transfers are virtual:
DRAM/HBM PageBackends reserve inaccessible address ranges (`PROT_NONE`), and the
transfer engine waits for modeled completion. Master/peer RPCs, background workers
and timers still use the real network and wall clock. Simulation is off by default.

## Enable and use

Keep the existing UMBP capacity, page size and DRAM/HBM policy configuration:

```python
from mori.umbp import UMBPClient, UMBPConfig

config = UMBPConfig.from_environment()
config.sim.enabled = True
config.sim.profile_path = "/path/to/profile.json"
config.sim.trace_path = "/tmp/umbp-sim-worker-0.jsonl"
client = UMBPClient(config)
client.put_from_ptr(key, ptr, nbytes)
client.get_into_ptr(key, ptr, nbytes)
```

Frameworks using `UMBPConfig.from_environment()` can instead set
`UMBP_SIM_ENABLE=1`, `UMBP_SIM_PROFILE=/path/to/profile.json` and
`UMBP_SIM_TRACE=/tmp/umbp-sim-worker-0.jsonl`. Existing put/get/batch/ranged APIs
and return types stay unchanged. Use one trace file per client/run.

For a local smoke run without a model, initialize `key`, `ptr`, and `nbytes`
before the example above (keep `buffer` alive until the client is destroyed):

```python
import ctypes
import mmap

key, nbytes = "smoke", 1024 * 1024
buffer = mmap.mmap(-1, nbytes)
ptr = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
```

Only effective DRAM/HBM backends are supported; SSD and standalone-process mode
are rejected. The default embedded backend is DRAM; `ssd.enabled` alone does not
select an SSD backend. Virtual GPU caller buffers need GPU location/device
metadata through the existing C++ `RegisterMemory` API. Real framework pointers
can use existing classification. A virtual HBM backend needs no GPU, but the
normal UMBP build dependencies, including HIP, still apply.

## Calibrate on the target platform

Calibration is offline: sample real transfers, fit a profile, then validate that
profile on separate workloads. The packaged sampler covers **CPU→CPU only**;
GPU/network coefficients require separate target-platform measurements.

```bash
cmake --build build --target umbp_sim_probe -j 8
python examples/umbp/calibrate.py \
  --probe build/src/umbp/umbp_sim_probe \
  --output /tmp/umbp-calibration \
  --cpus 4,5,6,7 --threads 1 2 4 \
  --cache cold --samples 24 --rounds 3 \
  --max-mib 256 --concurrency 1 2 4
```

Use a Release build, NumPy, Linux `taskset`, a new output directory and available
physical CPUs representative of deployment. Each N-thread probe uses the first
N listed CPUs and sets `UMBP_DRAM_COPY_THREADS` and `UMBP_DRAM_WRITE_THREADS` to N.
Sampling supports 1/2/4/8 threads and must include 1. Defaults use up to eight
allowed CPUs. Keep copy settings and CPU/NUMA placement consistent during simulation.
`--concurrency` defaults to 1/2/4 and must include 1. It adds a sparse validation
sweep of simultaneous calls sharing one client/engine, with disjoint keys and
buffers. Callers start together; thread creation is outside API timers. Each run
uses the first `min(N * callers, listed CPUs)` cores, so limited CPU lists can
oversubscribe. This tests concurrent reads and concurrent writes separately;
it does not simulate independent ranks/processes or simultaneous read/write mixes.
Use `--concurrency 1` to skip that sweep.

- `profile.json` loads directly through `UMBP_SIM_PROFILE`; only `cpu_cpu` is fitted.
- `report.json` contains platform/CPU identity, affinity, copy settings, profile/probe
  hashes, raw measurements and independent real/sim comparisons (median, p95, signed
  error). Tolerance is `max(2 µs, 10% of real median)`. High error is reported;
  failed operations or invalid fits fail the run.

Training measures real `LocalCopyEngine` Submit+Wait with 512 B–4 MiB blocks and
batches 1/8/32/64/128. Independent validation uses different block sizes from
1 KiB–8 MiB and batches 1/4/16/128 for transfers and `BatchPut`/`BatchGet` APIs.
Both reach 256 MiB aggregate payload by default. `--max-mib` bounds that payload
across all callers; actual memory also includes destinations and storage pools.
Increase it if necessary to exceed the target cache (the fixed grid tops out at
512 MiB training and 1 GiB single-caller validation).
Concurrent validation samples 8 KiB, 512 KiB and 2 MiB blocks with batches 1/16/128,
subject to the same aggregate cap. Small blocks represent copy sizes, not a full
framework's layer/ranged-call geometry. Only real single-caller training transfers
fit coefficients: API/control/queue time is not charged twice. Reports separate
concurrency levels. API timings include trace output to `/dev/null`, excluding Python.

`cold` flushes the caller source before each transfer/put, outside the timer
(x86 only); get reads the just-written store. `reuse` repeats buffers and caps
training at 8 MiB; larger validation working sets are flagged. Neither guarantees
cache hits or infers cache state. A reused-source profile can underestimate store
reads, so judge the **API** validation report as well as the transfer fit.

## Model and peer sharing

[sim_profile.json](../examples/umbp/sim_profile.json) provides illustrative,
uncalibrated values. Each configured path (`cpu_cpu`, `cpu_gpu`, `gpu_cpu`,
`gpu_gpu`, `remote_read`, `remote_write`) has:

| Field | Meaning |
| --- | --- |
| `channel` | Paths with the same name share a service queue within one client. |
| `latency_ns` | Nonnegative fixed overhead per submitted batch on this path. |
| `copy_latency_ns` | Optional nonnegative cost per coalesced copy; default 0. |
| `bandwidth_bytes_per_second` | Positive effective bandwidth. |

CPU coefficients may be arrays indexed by copy threads minus one, clamped to the
last entry; other paths use scalars. The simulator reuses LocalCopyEngine's
thread selection, byte threshold and background-thread restriction. CPU channel
capacity is the maximum calibrated bandwidth up to the larger configured copy/
write thread limit. Set those limits before creating the client.

```text
overhead = latency_ns[p] + copies * copy_latency_ns[p]
private_ns = overhead + bytes * 1e9 / bandwidth[p]
service_ns = overhead + bytes * 1e9 / channel_capacity
channel_available = max(submitted, channel_available) + service_ns
finish = max(submitted + private_ns, channel_available)
```

Model computation consumes the Submit time budget. Page fragmentation does not
multiply fixed latency; per-copy cost follows planner coalescing. Independent
submissions can overlap within channel capacity; paths sharing a channel sum
service costs. Zero-byte work adds no delay. Missing paths fail through normal
abort handling. This is an approximate queue model, without dynamic fair sharing.

For cross-peer sharing, run the normal Master and peers with distinct node IDs,
reachable addresses/peer ports and existing DRAM/HBM settings. Enable sim on
**every peer**, use a dedicated Master and add `remote_read`/`remote_write` to each
profile. For example, this entry under `paths` is illustrative, not measured:

```json
"remote_read": {
  "channel": "nic", "latency_ns": 8000,
  "bandwidth_bytes_per_second": 25000000000
}
```

Configure `remote_write` similarly. Directions are from the initiator; separate
channel names allow independent directions. Fit target-link **Submit+Wait**
measurements, not API totals or ping RTT. No automatic RDMA calibration is included.

Local copies complete in Submit; registered remote transfers complete in Wait.
Unregistered CPU callers use the real RDMA planner's bounce geometry with serial
CPU staging inside Submit. Pools, ranged scratch, bounce and queued re-cache
payloads are virtual. Ordinary/ranged load/offload and re-cache retain the real
control flow; a re-cached replica can serve other peers. Sim/real engines reject
each other's handshake, and no NIC registration or payload traffic occurs.

## Timing and limits

JSONL records contain `call_id`, `api`, `key_count`, requested `bytes` (range bytes
for ranged calls), monotonic `start_ns`/`elapsed_ns`, `result_count`, `exception`
and `profile_path`. Result count is the success/match count, or -1 for void calls
and exceptions. Records cover scalar/batch/depth/ranged put/get, existence queries,
register/deregister, clear, flush and close. IDs restart per client; completion
order may differ. Files append and flush on flush/close/destruction. Empty trace
paths use the INFO logger; disabling INFO then skips formatting.

`elapsed_ns` includes real control code, RPC/lock/queue waits and modeled transfers,
ending before trace serialization. Timing the whole API also includes trace output.
Transfers sleep until 100 µs before their deadline, then spin to reduce sleep
overshoot. Scheduling and control overhead still limit accuracy on short operations.
Independent background work is counted only when it delays an API or its channel.

- Channel queues are per client: they do not aggregate multiple initiators at a
  remote NIC or model switch congestion. Actual control RPC failures are retained;
  payload transport failures during modeled waits are not synthesized.
- Same-host peers exercise the protocol with that host's RPC cost, not a different
  cluster's latency. Profiles depend on platform, concurrency and workload.
- Virtual address ranges and page/key metadata still consume resources. Framework
  buffers, external copies and model execution remain real. With no restored payload,
  use trace/dummy execution; generated-token correctness cannot be validated.
- GPU/network accuracy and full SGLang performance validation remain separate work.

Build `test_umbp_sim` and run `ctest -R '^umbp_sim$'` for functional checks. Calibration
math checks are in `tests/python/umbp/test_sim_calibration.py`; they do not require GPUs.
