# Native L1 sharing across trays

Built on [MORI #1045](https://github.com/ROCm/mori/pull/1045), with
[ATOM #3828](https://github.com/ROCm/ATOM/pull/3828). The consumer imports the
owner's FABRIC HBM and copies directly into its held KV slots. gRPC carries
metadata; there is no staging buffer, lower tier, or persistent mapping cache.
Local sharing runs first; peers extend the prefix in windows of up to 128 keys.
Each window selects one peer and is pinned, copied and released before the next
window. Peer discovery stops at the first valid full-window hit. Fetch stops at
an incomplete window, an unrecovered read/copy failure, or an unconfirmed release,
returning the confirmed prefix unless completion is unknown.
Probe-only Fetch counts the same windowed prefix without acquiring read pins.
Limits are eight peers, 64 layout regions, and serialized mapping/copy operations.

Probe-only Fetch saves one first-remote-window hint per destination pool. Fetch
consumes it only when its current window fits the proved key prefix; AcquireRead
still resolves and pins current data. A stale or rejected hint can retry discovery
once, after confirming that no copy was submitted and the grant was released (or
the new owner rejected the old incarnation before granting). An unconfirmed
release or unknown copy completion never retries discovery or copying.
Drained ReleaseRead itself can retry in the background as described below.
Concurrent requests can replace
the hint, reducing its hit rate without bypassing read validation.

## Deployment

Both trays need HIP FABRIC GPUs in the same active vPOD/fabric domain. Workers,
services and peers must belong to the same trust domain: gRPC is insecure and
does not authenticate callers. Exported descriptors grant access to the whole
backing allocation, not just the slots in a read grant. Keep unrelated tenant
data out of that allocation and restrict endpoint access to trusted peers.
Current FABRIC mappings are read-write. Internal interface binding does not
authenticate callers; even client-side read-only mappings would not isolate
tenants because a descriptor holder controls its own mappings.

GPU ordinals and visibility must match between each worker and its service.
Install matching MORI Python and native packages on all workers and services
(`NATIVE_L1_FABRIC_VERSION=2`); #1045 alone lacks the FABRIC helper APIs.
The worker also requires the MORI-IO allocator and PyTorch MemPool support.
Only arenas allocated through MORI-IO `FabricMalloc` / `make_fabric_mem_pool`
are supported: one exportable allocation handle mapped at offset zero.
Arbitrary VMM arenas, including expandable segments or arenas assembled from
multiple allocation handles, are unsupported.
Use one compatible HIP runtime for PyTorch, MORI-IO and native L1 within each
worker process; mixing SDK core and devel libraries can break GPU discovery.

In a configured ROCm build, enable `BUILD_IO`, `BUILD_UMBP`, and
`BUILD_UMBP_L1_PYTHON`, with `GPU_TARGETS` set for the deployed GPUs, for example
`gfx1250` or `gfx942;gfx1250`. The build compiles each architecture separately
with `--no-gpu-bundle-output` to produce a loadable device ELF directly.

```bash
cmake --build /path/to/build --target mori_umbp_l1 mori_fabric_copy_kernel
export PYTHONPATH="/path/to/build/src/umbp${PYTHONPATH:+:$PYTHONPATH}"
# Tray A, using example internal addresses and gfx1250 GPUs.
python tools/umbp_l1_fabric_server.py --listen 10.0.0.10:29620 \
  --peer 10.0.0.11:29620 --kernel /path/to/build/lib/fabric_copy.gfx1250.hsaco
# Tray B.
python tools/umbp_l1_fabric_server.py --listen 10.0.0.11:29620 \
  --peer 10.0.0.10:29620 --kernel /path/to/build/lib/fabric_copy.gfx1250.hsaco
```

Adjust addresses and paths to the deployment. Select the local GPU architecture's
`lib/fabric_copy.<arch>.hsaco` explicitly with `--kernel`; manual unbundling is
unnecessary. FABRIC mode requires the kernel and never falls back to HIP memcpy.
The service is L1-only; no tier server is needed.

Configure ATOM `umbp_pool` with `transport: "fabric"`, the local service
`address`, a unique `node_id` per tray, and a common explicit `group`.
Each engine needs a distinct namespace. Group members must match the model
revision, KV/hash conventions, block size, layer/shard layout, and speculative
configuration; equal region sizes alone are insufficient. IPC remains the default.

## Ownership and shutdown

`NativeL1Memory` allocates and zeroes the final KV arena inside the worker and
owns its allocator, descriptor, registration, and Detach. ATOM supplies layout,
slots, and write progress. Keep the helper alive while any arena view is in use:

Use `allocate(nbytes, device)` and `bind_layout(arena, num_slots, regions)`;
each region is a `(byte offset, unit size)` pair. Stop admission/local writes
and confirm peer drain before `close()` or releasing views.

Owner Commit confirms writes; AcquireRead pins current source slots and
destination writers prevent reuse. Incarnation and operation IDs fence stale
or replayed requests. ReleaseRead unpins only after confirmed completion or
no submitted work. Full descriptor identity, range and allocation-level alias
checks remain enforced despite descriptor deduplication and copy coalescing.

Unknown completion retains mappings, pins, destinations, and the worker's arena;
ATOM stops serving after a FABRIC Fetch exception. A failed `close()` retains
memory and requires Detach from the original service incarnation. Propagate
that failure and keep owners/services alive until drain is confirmed. The
busy-pool wait is bounded to two seconds; timeout never permits memory reuse.
Python references cannot protect HBM after process exit.

Read admission uses owner-issued epochs. After an epoch ends, delayed AcquireRead
requests from it are rejected and released receipts can be reclaimed; active
grants remain pinned until ReleaseRead confirms drain. Each service retains at
most 65,536 active receipts and current-epoch tombstones.

After confirmed drain or no submitted work, a failed ReleaseRead stops that Fetch
and is retried in the background with bounded pending records and capped backoff.
Retries end on acknowledgement or owner incarnation change. Unknown completion
is never released automatically; consumer crashes still require coordinated
drain and recovery.
`UMBP_L1_KEEP_COPIES` is local to each PoolManager.
Remote copies log `[L1Peer] fabric completed` and count at the destination;
source `served_out_slots` counts local copies only.
