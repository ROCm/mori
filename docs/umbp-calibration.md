# UMBP transfer calibration

The optional probes train profiles with real engines, then compare independent
real/sim workloads. Use the same build for the probes and the
[UMBP simulator](umbp-simulation.md).

```bash
cmake --build build --target umbp_sim_probe umbp_sim_remote_probe -j 8
python examples/umbp/calibrate.py \
  --probe build/src/umbp/umbp_sim_probe --output /shared/calibration \
  --paths all --device 0 --threads 1 4 --cpus 4,5,6,7 --ssd-dir /data/calibration \
  --ssd-io io_uring --ssd-read-prefill-mib 256 --ssd-api-settle-ms 250 \
  --local-address 10.0.0.1 --remote-address 10.0.0.2 \
  --remote-command 'ssh node2 /path/to/build/src/umbp/umbp_sim_remote_probe'
```

The default is CPU-only. For local GPU calibration, select
`--paths cpu_cpu cpu_gpu gpu_cpu gpu_gpu` and omit remote settings.
Both remote nodes need working RDMA, the same build and access to the output/profile
path. Use reachable IPv4 addresses. The driver appends quoted owner arguments to
`--remote-command` and records `owner.log`; an existing SSH/container launcher must
provide the environment. The driver does not deploy images or allocate nodes.
GPU/RDMA callers use the full `--cpus` set to allow CQ polling alongside submission.
The owner launcher receives that affinity; forward it or select suitable remote CPUs.

Native probes accept `--block-bytes B,...` and `--batches N,...`. The remote probe
also accepts `--gap-bytes` (default 64) and `--staged`; local staged SSD API probes
accept `--gpu-caller`. The driver selects these modes automatically during validation.

| Path | Real engine and validation |
| --- | --- |
| `cpu_cpu` | `LocalCopyEngine`; engine and UMBP put/get. |
| `cpu_gpu`, `gpu_cpu` | `HbmCopyEngine`, including registration and gather/fallback; DRAM→GPU load and GPU→DRAM offload. |
| `gpu_gpu` | `HbmCopyEngine` and HBM-tier put/get on one selected GPU. |
| `remote_read`, `remote_write` | `MoriIoEngine` with registered CPU DRAM endpoints; direct and staged engine validation. |
| `cpu_gpu_gather`, `gpu_cpu_gather` | Actual HbmCopyEngine gather kernels, fitted separately from memcpy. |
| `cpu_staging`, `cpu_staging_read` | Serial copyin/copyout on actual MoriIo bounce buffers; RDMA refresh precedes copyout timing. |
| `file_cpu`, `cpu_file` | `PosixEngine` on real SSD record extents; fresh write reservations. |
| `file_gpu` | `GdsEngine` reads with hipFile registration; requires GDS support, without simulated fallback. |
| `file_cpu_staged`, `cpu_file_staged` | `SSDTier` payload I/O, record/CRC/bounce work and durability; engine and CPU/GPU API validation. |
| `file_cpu_staged_no_crc` | SSDTier reads without CRC; automatically trained and engine-validated alongside CRC reads. |

Real probes verify complete payloads; sim uses inaccessible virtual buffers.
Allocation, registration, planning, handshake, eight warmups and payload checks stay
outside Submit+Wait timers. API timers include actual control work and trace output
to `/dev/null`; API/control observations never enter the transfer fit. Remote API
control timing is available separately through the simulator trace. `gpu_gpu` follows
HbmCopyEngine's host return without extra device synchronization, so its fitted
bandwidth does not measure device-drain throughput.

## Sampling

| Option | Behavior |
| --- | --- |
| `--samples`, `--rounds` | Defaults: 24 and 3. |
| `--probe-timeout` | Default: 300 seconds per invocation; increase for large grids or settle delays. |
| `--train-block-bytes`, `--train-batches`, `--validation-block-bytes`, `--validation-batches` | Replace default grids; positive entries only, sampled TRAIN/heldout overlap rejected. |
| `--max-mib` | Default: 256 MiB aggregate workload across callers, including SSD record padding or remote gaps. Both probes honor this cap. |
| `--threads` | CPU copy-thread curves; GPU/network/staging fits remain scalar. |
| `--concurrency` | Default: 1/2/4; heldout validation only. Callers share an engine and use disjoint buffers. |
| `--cache cold\|reuse` | Local host-source cache condition; default `cold` displaces data before operations. GPU caches are not flushed; RDMA follows its round-trip workload. |

Keep CPU/NUMA/GPU placement and cache conditions consistent when reusing profiles.
Cold flushing requires x86. Small reused buffers can overestimate DRAM bandwidth;
API staging may still warm internal sources, so engine/API cache conditions can differ.

## SSD configuration and preparation

Use an explicit writable `--ssd-dir` on the SSD being calibrated, not NFS/tmpfs.
Probes remove only their own temporary child directories. Match deployment settings:

- `--ssd-io posix|io_uring`: probe default `posix`, UMBP default `io_uring`.
  Unavailable io_uring, including seccomp restrictions, fails explicitly.
- `--ssd-durability relaxed|strict`: default `strict`, matching UMBP.
- `--ssd-no-crc`: staged-read rows use `file_cpu_staged_no_crc`; an unmeasured CRC
  curve remains an estimate. Without this flag, both read curves are sampled.
- SSD payload sampling uses the default four `tier_io_threads`; `--threads` controls
  CPU copies. API GPU staging uses HBM curves, which should also be calibrated.

`--ssd-read-prefill-mib` defaults to zero. A positive value writes unrelated records
outside timing after seeding each read fixture, with matching preparation for TRAIN
and heldout engines. API reads then retain seeded keys and repeat only loads. Fixtures
reserve storage/staging for preparation. Choose prefill size for the device; it can
displace recent writes but does not guarantee cold caches or identical fresh extents.

`--ssd-settle-ms` defaults to 10 ms before engine reads. `--ssd-api-settle-ms` defaults
to zero for immediate reload. Retained loads may reuse DRAM staging leases; with the
usual 200 ms lease reaper, use 250 ms to validate repeated storage reads, and check
that loads reach storage. Adjust for deployment's reaper interval. Both waits are
outside timers and do not flush device caches. Report reload, staging-hit and retained
storage-read conditions separately; they are recorded with prefill in `report.json`.

Staged TRAIN timing includes complete payload work and write-buffer cleanup, excluding
index, locks and planning. `wall_elapsed_ns` keeps whole-call time, `model_bytes` keeps
padded I/O bytes and `segments` keeps actual geometry. Engine/API validation uses whole
wall time. GDS first-FD registration fits separately from TRAIN `api=register` rows and
is validated with the same lifecycle; `registration_latency_ns` applies once per live FD.

## Profiles and accuracy

Only real, single-caller TRAIN data fits nonnegative fixed overhead, per-copy/segment
cost and effective bandwidth. CPU thread curves interpolate in time/byte. Include small
and deployment-sized large blocks, with independent validation shapes. Unresolved byte
cost keeps the measured copy cost and uses the largest finite bandwidth, reported in
`zero_byte_cost_paths`; this does not identify physical bandwidth.

- GPU TRAIN separates gather/memcpy using actual kernel launches. Sim applies production
  registration/geometry/cutoff rules and falls back to base curves when variants are
  absent. Measured H2D/D2H base curves set `interleave_copies=true` for blocking per-copy
  fallback; old profiles retain batch scheduling.
- Directional staging can fit `max(0, bytes - size_knee_bytes) * large_size_ns_per_byte`
  only with identifiable TRAIN sizes on both sides. Staging sampling follows timed wire
  transfers, and read ranges are cleared before RDMA refresh. Older reads fall back to
  `cpu_staging`, then serial `cpu_cpu`. Model calculation stays inside stage budgets.
- SSD writes fit combined preparation/I/O/durability/cleanup. Reads optionally fit
  nonnegative per-I/O `log_size_latency_ns` and per-batch `batch_log_size_latency_ns`,
  multiplying log physical bytes per I/O. Unidentifiable or zero terms are omitted.
  Native SSDTier reads emit `serialize_latency=true` for their I/O mutex. Compatible
  no-CRC curves reserve queued I/O on the shared channel; remaining CRC work completes
  outside that reservation. Missing/incompatible curves and mixed submits retain full
  payload service cost; writes serialize service. Older staged paths use direct curves.

`--base-profile` defaults to complete example estimates. Measured paths replace entries;
unmeasured paths remain estimates, listed in `path_sources`. Custom bases must include
unmeasured/reverse API paths. `profile.json` identifies measured paths and estimate count;
`report.json` retains raw data, settings, profile/base hashes, medians, signed errors and
passes at `max(2 µs, 10% of real median)`. Operation, fit or isolation failure prevents
publishing a completed profile; large timing errors remain visible in a successful run.

Concurrent RDMA uses an approximate serialized channel model; independent processes/ranks,
GPU RDMA endpoints, inter-GPU P2P and distributed framework API totals are not calibrated.
Gather validation uses one GPU/direction per submit; mixed batches aggregate fixed costs
per direction. SSD models do not track device cache/placement/post-write state, queue
saturation or preparation/cleanup overlap; mixed read/write submits are not independently
calibrated. Check heldout errors for the actual deployment before reusing parameters.

Run fitting/isolation tests with NumPy and pytest:

```bash
python -m pytest -q tests/python/umbp/test_sim_calibration.py
```

Hardware accuracy is verified with actual probes, not generated-token correctness.
