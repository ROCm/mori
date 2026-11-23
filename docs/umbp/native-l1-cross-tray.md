# Native L1 sharing across trays

This opt-in extension builds on [PR #1045](https://github.com/ROCm/mori/pull/1045)
at `fb01cb16bdbdcfb21ebedbda4c89fb226cd82161` (open on 2026-11-23).
The base was rebased from `82996281`; its L1 changes and this PR's implementation
are unchanged. Validation revisions below identify the original tested commits.
It uses the FABRIC mapping/copy approach from PR #1024 and the KV-sharing branch:

```text
ATOM owner KV HBM → imported FABRIC view → copy kernel → ATOM consumer KV slots
```

The consumer reserves its final slots, acquires a peer read grant that pins the
source, and copies directly into those slots. gRPC carries metadata only. There
is no staging allocation, DRAM/SSD tier, master, or PoolClient routing.
Local sharing is checked first; each Fetch can extend that continuous prefix with
up to 128 keys from one peer. Limits are eight peers and 64 layout regions;
copies and FABRIC mapping operations are currently serialized.

## Deployment

Both trays require HIP FABRIC GPUs in the **same active vPOD/fabric domain**,
reachable L1 endpoints, and this matching MORI Python package/native extension.
The worker also needs the regular MORI-IO allocator and PyTorch MemPool support.
The extension exposes `NATIVE_L1_FABRIC_VERSION=1`; #1045 alone lacks these APIs.
The service is pure L1 and does not require the standalone tier server.

In a configured ROCm build, enable `BUILD_IO`, `BUILD_UMBP` and
`BUILD_UMBP_L1_PYTHON`, and set `GPU_TARGETS` (`gfx1250` on 455):

```bash
cmake --build /path/to/build --target mori_umbp_l1 mori_fabric_copy_kernel
export PYTHONPATH="/path/to/build/src/umbp${PYTHONPATH:+:$PYTHONPATH}"
```

The kernel is `/path/to/build/lib/fabric_copy.hsaco`. Adjust the extension path
if CMake output directories differ. Install the matching Python package as well:
building the extension does not install `mori.umbp_l1.NativeL1Memory` or the allocator.

Start one service per tray with reciprocal peers:

```bash
# Tray A
python tools/umbp_l1_fabric_server.py --listen 0.0.0.0:29620 \
  --peer TRAY_B_IP:29620 --kernel /path/to/build/lib/fabric_copy.hsaco
# Tray B
python tools/umbp_l1_fabric_server.py --listen 0.0.0.0:29620 \
  --peer TRAY_A_IP:29620 --kernel /path/to/build/lib/fabric_copy.hsaco
```

Use a trusted cluster network for the insecure gRPC endpoints. GPU ordinals and
visibility must match between workers and their local service. FABRIC mode
requires the kernel and never falls back to HIP memcpy or staging.

The companion ATOM adapter builds on [PR #3482](https://github.com/ROCm/ATOM/pull/3482)
at `db27ec2198711f2d77dafff8ed0d5621af59d9fb`. Configure `umbp_pool` with
`transport: "fabric"`, `address: "127.0.0.1:29620"`, a unique `node_id` per tray,
and a common explicit `group`; each engine on a tray needs a distinct namespace.
Group members must match model revision, KV/hash conventions, block size,
layer/shard layout and speculative configuration. Equal region sizes are not
enough. Local deployments retain the default IPC mode.

The helper runs inside the worker, allocates and zeroes the final KV arena, and
owns its allocator, descriptor, registration and Detach lifecycle. ATOM supplies
layout and write progress. Keep the helper alive while any arena view is in use:

```python
from mori.umbp_l1 import NativeL1Memory

memory = NativeL1Memory(client, pool_id, group)
arena = memory.allocate(nbytes, device)
memory.bind_layout(arena, num_slots, regions)  # byte offset/unit size pairs
# Use the arena and publish/commit through the existing L1 client.
# Stop admission/writes and confirm peer drain before releasing views.
memory.close()
```

## Completion and shutdown

AcquireRead resolves and pins current source slots; owner Commit establishes
write completion. Destination writers prevent overlapping Fetch/publication.
ReleaseRead unpins only after confirmed copy completion or no submitted work.
Incarnation and operation IDs fence stale or duplicated requests.

Unknown completion retains mappings, pins and destinations and quarantines the
pool. ATOM stops serving on a FABRIC Fetch exception. The helper retains arena
and allocator references before Attach and requires Detach confirmation from
that same service incarnation. A busy-pool lifecycle wait is bounded to two
seconds; failure raises without releasing memory. Propagate failed `close()`;
keep owners and services alive until peer drain and Detach are confirmed.
A timeout never authorizes memory reuse; Python references cannot protect HBM
after process exit.

There is no automatic recovery, Release retry or receipt reclamation. Each
service retains at most 65,536 receipts/tombstones per lifetime; plan a coordinated
drain/restart before exhaustion. Lost Release may leave source slots pinned.
`UMBP_L1_KEEP_COPIES` remains local to each PoolManager, with pins/writers also
protecting its kept list; it does not maintain a global cross-tray copy count.
Remote copies are logged as `[L1Peer] fabric completed` and counted at destination;
source `served_out_slots` counts local copies only.

## Validation scope

The 2026-11-23 run at MORI `965a6061` on g13-12 ↔ g13-10 passed a fresh native
extension/gfx1250 kernel build, 57 C++ pool/peer/RPC tests, copy planner,
19 peer tests with `KEEP_COPIES=1`, 12 Python helper tests and 134 companion
ATOM tests (one intentional native fault-injection skip).

Real GPU checks passed both import orders, all 32 directed device paths
(160 cases: bytes/guards/final pointer, prefix cap, pending writes and ATOM
adapter), and 16 GPU Event/Commit cases on four bidirectional device pairs.
DeepSeek-V4.1-Flash Fetch-off/on runs on the same consumer matched inputs,
complete output text and token counts for all 64 requests per arm including
warmup. Each of the 16 measured remote hits copied 96 slots / 18,309,120 bytes;
true misses and local-hot requests had no remote copy. Each node used attention
DP4/TP1 and MoE EP4, without cross-node TP. All three model instances confirmed
normal Detach before services stopped.

An earlier 2026-11-22 run also covered slot reuse and a ten-minute bidirectional
soak. These are bounded workload and coordinated-shutdown results, not full
model quality, saturated throughput, crash recovery or other topologies.

## Copy bandwidth tuning

The multi-region copy budget now matches MORI-IO's 1536 blocks; the single-region
path is unchanged. On g13-12 ↔ g13-10, three uninstrumented AB/BA/AB rounds using
the V4.1 16-region layout (128 keys / 24,412,160 bytes) improved all 60 paired
large-request Fetch windows: dense +6–22%, fragmented +13–42%. Each GPU had one
outstanding Fetch; these are repeated-address payload rates, not wire bandwidth
or model throughput. Small requests had mixed changes, including regressions;
longer windows did not show a consistent backend slowdown. No general
no-regression claim is made.

The tuning passed 57 C++ tests, the copy planner, KEEP_COPIES peer coverage and
180 GPU correctness cases, including all 32 directed paths, the ATOM adapter,
unaligned chunk tails and slot reuse. The model validation above predates this
budget-only change. Profiling after tuning puts AcquireRead metadata/RPC at
47–57% of large-request handler time.

AcquireRead now negotiates a per-response dictionary of full descriptors;
new and old services retain the inline-handle fallback. For the V4.1 layout,
an offline protobuf fixture reduces a 128-key response from about 332 KB to
25 KB. The receiver also coalesces contiguous source/destination runs within
each layout region before backend validation and copy planning: dense 128-key
requests become 16 fragments instead of 2048. Each original range remains
validated; descriptor identity, pin/writer protection and allocation-level
alias checks are preserved. No mapping cache or staging buffer is added.

Three rotated baseline/dictionary/combined rounds (540,000 timed Fetches) on
the same trays improved all 60 large-request windows versus `be244261`.
Single-worker dense rates rose from 15.6–17.0 to 25.5–26.0 GiB/s; fragmented
rates rose from 15.1–15.7 to 21.3–21.9 GiB/s. Four-pair bidirectional rates
improved about 25%/30% per worker. Coalescing adds 9–20% for single-worker dense
requests over dictionary-only, but its additional concurrent gain is mixed,
with small p99 regressions versus dictionary-only. Tiny requests also fluctuate;
three longer reverse windows changed −0.18%/+5.13%/−2.53%. These are bounded
repeated-address Fetch results, not a universal no-regression claim.

The combined extension passed 74 C++ tests, the planner and 27 KEEP_COPIES
peer tests, all 32 directed GPU paths, tails/slot reuse, the real ATOM adapter,
16 Event/Commit cases and bidirectional mixed-version service checks. ATOM and
the copy kernel are unchanged; the model run above predates these optimizations.
