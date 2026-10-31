# UMBP transfer calibration

Build the optional probes and run real-engine training followed by independent
real/sim validation. Generated profiles work with the [UMBP simulator](umbp-simulation.md).
Use the updated build for the optional gather, directional staging and SSD
payload models.

```bash
cmake --build build --target umbp_sim_probe umbp_sim_remote_probe -j 8
python examples/umbp/calibrate.py \
  --probe build/src/umbp/umbp_sim_probe --output /shared/calibration \
  --paths all --device 0 --threads 1 4 --cpus 4,5,6,7 --ssd-dir /data/calibration \
  --ssd-io io_uring --ssd-read-prefill-mib 256 --ssd-api-settle-ms 250 \
  --local-address 10.0.0.1 --remote-address 10.0.0.2 \
  --remote-command 'ssh node2 /path/to/build/src/umbp/umbp_sim_remote_probe'
```

Both nodes need the same build, working RDMA and access to the output/profile path.
The remote command includes the owner probe executable; the driver appends quoted owner
arguments. It can include an existing container launcher. It does not deploy images,
request nodes or copy profiles. Use explicit reachable IPv4 addresses, not hostname
resolution that may select loopback. Owner logs append to `owner.log`.
GPU/RDMA caller probes use the full `--cpus` set so a CQ poller does not compete
with the submitter on one core. The owner launcher receives that local affinity;
an SSH/container launcher must forward it or explicitly set a suitable remote CPU set.

For a local run, omit remote settings and select `--paths cpu_cpu cpu_gpu gpu_cpu gpu_gpu`.
The default remains CPU-only; `--paths cpu_cpu` needs no GPU or remote host.

| Path | Real engine and validation |
| --- | --- |
| `cpu_cpu` | `LocalCopyEngine`; independent engine and UMBP put/get timings. |
| `cpu_gpu`, `gpu_cpu` | `HbmCopyEngine`, including host registration and production gather/fallback; DRAM→GPU load and GPU→DRAM offload APIs. |
| `gpu_gpu` | `HbmCopyEngine` and HBM-tier put/get on the selected GPU. This does not measure inter-GPU P2P. |
| `remote_read`, `remote_write` | `MoriIoEngine` with registered CPU DRAM endpoints; direct and staged engine validation. GPU RDMA endpoints are not sampled. |
| `cpu_gpu_gather`, `gpu_cpu_gather` | Optional curves for the actual HbmCopyEngine gather kernel; base direction curves fit memcpy samples. |
| `cpu_staging`, `cpu_staging_read` | Serial copyin/copyout on the actual MoriIo bounce buffer; copyout sources are refreshed by RDMA before timing. |
| `file_cpu`, `cpu_file` | `PosixEngine` reading/writing real SSD record extents; writes use fresh reservations. |
| `file_gpu` | `GdsEngine` file→GPU reads with hipFile registration and complete GPU payload checks. Requires GDS support; no simulated fallback. |
| `file_cpu_staged`, `cpu_file_staged` | Actual `SSDTier` payload I/O, record/CRC/bounce work and configured durability. Independent API validation covers CPU and GPU callers. |
| `file_cpu_staged_no_crc` | The same SSDTier read backend without CRC; automatically trained and engine-validated with `file_cpu_staged` when CRC is enabled. |

Real runs verify complete payloads. Sim uses inaccessible virtual buffers. Allocation,
registration, planning, control handshake, eight warmups and payload checks are outside
transfer timers; transfer timing is Submit+Wait. API timings include the real API control
work and trace output to `/dev/null`. The fit never includes API/control overhead twice.
Remote API/control timing can be measured separately through the simulator API trace.
`gpu_gpu` measures the engine's host return boundary. Its HIP copy fallback has no
additional device synchronization; a small return time or zero fitted byte cost
does not identify GPU copy throughput or subsequent device-drain time.

Bounce-buffer copy curves can fit a second slope after an aggregate byte threshold:
`max(0, bytes - size_knee_bytes) * large_size_ns_per_byte`. This accounts for slower
large DRAM copies. Both fields are selected only from real single-caller training,
with identifiable sizes on both sides; unchanged slopes keep the original linear model.

SSD sampling requires an explicit writable `--ssd-dir` on the device being calibrated,
not NFS or tmpfs. The probe creates and removes only its own temporary child directory.
`--ssd-io posix|io_uring`, `--ssd-durability relaxed|strict` and `--ssd-no-crc`
select the real tier configuration; durability defaults to `strict`, matching UMBP.
The probe defaults to `posix`, whereas UMBP defaults to `io_uring`; select the same
backend as the deployment when calibrating staged I/O. Unavailable `io_uring` fails
explicitly, including when a container's seccomp policy blocks it.
SSD payload measurements use the configuration's default four `tier_io_threads`;
`--threads` controls CPU copy-engine curves, not these SSD threads.
Reuse a profile with the same settings.
For retained L3 data, `--ssd-read-prefill-mib 256` writes unrelated KV records after
seeding each read fixture, outside timing. Training and held-out engine reads use
the same preparation. API read validation seeds once, fills the tier, then repeats
only loads without rewriting or clearing KV. The fixture reserves enough storage
and staging space for preparation; real and sim use the same configuration.
The required prefill size depends on the device. This can displace recent writes
from a fast device state, but it is not a universal cache-flush guarantee. The
report records the prefill size and API read mode.
`--ssd-settle-ms` (default 10 ms) waits after synchronizing read fixtures, outside
engine timing. This does not flush the SSD controller's caches or guarantee cold
reads. Small fixtures repeatedly access the same extents; fresh fixtures can have
different read latencies even after long warmups. Keep these conditions consistent
when comparing or reusing profiles.
With the default prefill of zero, API validation keeps immediate reload. Retained-key
loads can reuse DRAM staging leases. To validate repeated storage reads with the
default 200 ms lease reaper, use `--ssd-api-settle-ms 250`; adjust this delay if the
deployment changes the reaper interval, and check that loads reach storage. The wait
is outside API timing and recorded in the report. It does not flush the device cache.
Report immediate reload, staging hits and retained storage reads separately. The
model does not track device cache contents, physical placement or post-write read state.
Staged training measures complete payload work, excluding the index, locks and planning;
`wall_elapsed_ns` retains whole-call time for audit. `model_bytes` records padded I/O
bytes and `segments` records actual copy/I/O geometry. Validation compares whole engine
or API wall time, including the unchanged control work. API GPU staging uses the same
platform's HBM curves; calibrate those paths too rather than silently relying on estimates.
GDS first-FD registration is timed separately from Submit+Wait and fitted only from
training `api=register` rows. Held-out registration rows compare the same lifecycle;
cached registrations retain their ordinary bookkeeping cost. The optional
`file_gpu.registration_latency_ns` is charged once per live FD registration.

## Sampling and profiles

- `--samples` and `--rounds` default to 24 and 3.
- `--probe-timeout` defaults to 300 seconds per invocation; increase it for large
  grids or long API settle delays.
- `--train-block-bytes`, `--train-batches`, `--validation-block-bytes` and
  `--validation-batches` replace the default grids. All entries must be positive;
  sampled training/validation overlap is rejected.
- `--max-mib` defaults to a 256 MiB workload footprint across callers. SSD probes
  include record padding and remote probes include their 64-byte gaps.
- `--threads` selects CPU copy-thread curves; GPU/network/staging fits are scalar.
  `--concurrency` defaults to 1/2/4 and affects validation only. Callers share one engine
  with disjoint buffers; independent process/rank contention is not fitted.
- `--cache cold|reuse` applies to local host-source copying. GPU caches are not flushed;
  remote cache state follows the round-trip workload. Keep CPU/NUMA/device placement
  and cache conditions consistent when reusing a profile. Both modes honor `--max-mib`;
  small reused buffers can stay in CPU cache and overestimate DRAM bandwidth for KV
  staging. Use the default `cold` for data displaced from cache between operations.
  This controls the caller's source, not all internal buffers: API offload copies
  into staging before SSD record preparation, which can warm that stage's source.
  Direct engine and API cache conditions can therefore still differ.

Only real, single-caller training samples fit nonnegative fixed overhead, cost per
actual copy/segment and effective bandwidth. CPU thread counts interpolate in time/byte.
Staged-write fitting uses the combined measured time for record preparation,
I/O, configured durability and record-buffer cleanup. Include small blocks to
retain fixed write/sync overhead and deployment-sized large blocks to fit byte
cost; use different sizes/batches for validation. The basic model does not separate
preparation or cleanup overlap between callers.
SSD read fits also allow nonnegative `log_size_latency_ns` per I/O and
`batch_log_size_latency_ns` per batch. Both multiply the log of physical bytes per
I/O: the first describes serial reading, the second a shared completion latency
for concurrent native reads. Training selects the coefficients without hard-coding
the I/O backend. Terms are omitted when the grid cannot distinguish them or the fit
selects zero. They do not model device cache state or queue saturation; held-out
validation remains necessary for the selected backend and batch sizes.
The fit allows zero size cost when training cannot resolve it. This is encoded
as the largest finite effective bandwidth and listed in `zero_byte_cost_paths`;
it does not identify physical bandwidth. This preserves the measured per-copy
cost instead of dropping that model because of a slightly negative noisy size slope.
GPU training separates gather from memcpy by observing actual kernel launches.
Sim selects the optional gather curve using registered host ranges, per-device
segment geometry and the production cutoff; missing variants fall back to the base curve.
Measured base `cpu_gpu` / `gpu_cpu` curves set `interleave_copies=true` to model
HbmCopyEngine's blocking per-copy fallback. Older profiles keep batch scheduling.
Direct RDMA trains network paths; its actual bounce buffers train staging in each direction.
Both probes honor the explicit `--max-mib` limit; remote sampling has no hidden 64 MiB cap.
Staging samples run after the timed wire pair so extra calibration traffic does
not warm the following network sample. Read bounce and destination ranges are
cleared before RDMA refresh, keeping payload validation independent of earlier copies.
Held-out staged transfers verify the combined model without fitting network costs twice.
Concurrent direct RDMA uses a serialized channel approximation; real QPs can share
bandwidth differently, especially for large transfers. Check concurrent validation
before reusing single-caller parameters.
Old profiles can omit `cpu_staging_read`: reads fall back to `cpu_staging`, then
the serial `cpu_cpu` curve. Each stage includes model calculation in its time budget.

`--base-profile` defaults to the simulator's complete example estimates. Measured paths
replace those entries; unmeasured entries remain estimates. `profile.json` therefore
loads for all configured example paths even after CPU-only sampling. Its ID identifies
measured paths and the number of estimated entries; `report.json` lists `path_sources`.
A custom base must include the workload's unmeasured/reverse API paths.
Unselected SSD paths remain estimates. Staged curves are optional: older profiles use
the corresponding direct file curve. Separate curves avoid applying record/CRC overhead
to direct file reads that bypass those steps.
CPU SSD API validation runs once through the staged path names; direct CPU paths
validate their engines. Independent SSD reads share modeled bandwidth while their
fixed I/O latency can overlap. Native SSDTier reads emit `serialize_latency=true`
because its I/O mutex covers batch submission and completion. With a matching
serialized no-CRC curve on the same channel, its cost estimates queued I/O; the
nonnegative remainder of the CRC curve finishes afterward without holding the
queue. Missing or incompatible no-CRC curves and mixed read/write submits retain
the full payload service cost. Writes retain serialized channel service.
Mixed read/write submit batches are not independently calibrated.

`report.json` retains raw measurements, sampling settings, profile hash, per-path
real/sim medians, signed errors and tolerance results. Tolerance is
`max(2 µs, 10% of real median)`. High error is reported; operation, fit or validation
isolation failures prevent publishing a completed profile. A successful workflow is
not an all-cases accuracy pass. Gather thresholds, NUMA, concurrency and transfer-size
regimes can require more detailed models; these probes expose those errors.
Gather accuracy is validated on one selected GPU and one direction per submit.
Mixed-direction and multi-GPU gather batches retain functional support, but their
fixed launch costs are aggregated per direction and are not independently calibrated.

Run the fitting/isolation tests with:

```bash
python -m pytest -q tests/python/umbp/test_sim_calibration.py
```

These tests need NumPy and pytest. Hardware accuracy is verified using the actual probes,
not by unit-test timing assertions or generated-token correctness.
