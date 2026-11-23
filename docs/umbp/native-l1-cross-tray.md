# Native L1 sharing across trays

Built on [MORI #1045](https://github.com/ROCm/mori/pull/1045), with
[ATOM #3717](https://github.com/ROCm/ATOM/pull/3717). The consumer imports the
owner's FABRIC HBM and copies directly into its held KV slots. gRPC carries
metadata; there is no staging buffer, lower tier, or persistent mapping cache.
Local sharing runs first; one peer can extend the prefix by up to 128 keys.
Limits are eight peers, 64 layout regions, and serialized mapping/copy operations.

## Deployment

Both trays need HIP FABRIC GPUs in the same active vPOD/fabric domain and
reachable endpoints on a trusted network (gRPC is insecure). GPU ordinals and
visibility must match between each worker and its service. Install matching
MORI Python and native packages; #1045 alone lacks the FABRIC helper APIs.
The worker also requires the MORI-IO allocator and PyTorch MemPool support.

In a configured ROCm build, enable `BUILD_IO`, `BUILD_UMBP`, and
`BUILD_UMBP_L1_PYTHON`, with `GPU_TARGETS` set for the GPUs (`gfx1250` on 455):

```bash
cmake --build /path/to/build --target mori_umbp_l1 mori_fabric_copy_kernel
export PYTHONPATH="/path/to/build/src/umbp${PYTHONPATH:+:$PYTHONPATH}"
# Run on each tray with the other tray's address.
python tools/umbp_l1_fabric_server.py --listen 0.0.0.0:29620 \
  --peer OTHER_TRAY_IP:29620 --kernel /path/to/build/lib/fabric_copy.hsaco
```

Adjust paths to the build outputs. FABRIC mode requires the kernel and never
falls back to HIP memcpy. The service is L1-only; no tier server is needed.

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

There is no automatic recovery, Release retry, or receipt reclamation. Each
service retains at most 65,536 receipts/tombstones; coordinate drain/restart
before exhaustion. `UMBP_L1_KEEP_COPIES` is local to each PoolManager.
Remote copies log `[L1Peer] fabric completed` and count at the destination;
source `served_out_slots` counts local copies only.

Hardware validation on g13-12 ↔ g13-10 covered all 32 directed GPU paths,
full bytes/guards, slot reuse, Event/Commit, and the ATOM adapter. Model E2E
predates performance optimizations; fixed-address microbenchmarks do not
establish model throughput, physical link bandwidth, saturation, or crash recovery.
