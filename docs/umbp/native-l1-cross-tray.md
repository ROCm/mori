# Native L1 sharing across trays

Based on [PR #1045](https://github.com/ROCm/mori/pull/1045) at
`82996281c5800ec63541cff7c793e7f86fc9907c` (open on 2026-11-23).
This opt-in extension is required in addition to #1045 for cross-tray sharing. It uses
the FABRIC mapping/copy approach referenced by PR #1024 and
`feat/umbp-ualink-kv-sharing`, without merging their general pool/tier stacks.

## Scope and data path

```text
ATOM owner native KV HBM → imported FABRIC view → copy kernel → ATOM consumer native KV slots
```

The consumer reserves its final slots first. A local L1 service probes static
peers, acquires a read grant that pins the owner's current source slots, imports
their FABRIC descriptors, and copies directly into those destinations. Only
address/length metadata travels over gRPC. There is no KV payload staging,
separate KV allocation, DRAM/SSD tier, master, or PoolClient routing.

Local sharing remains first in lookup order. The initial remote path selects one
peer for up to 128 following keys per Fetch; it can return a shorter continuous
prefix. Up to eight peers and 64 layout regions are supported. Copies and FABRIC
mapping operations are serialized in this first version. This is a correctness
implementation, with no throughput claim.

## Integration

Both trays need supported HIP FABRIC GPUs in the **same active vPOD/fabric domain**,
the matching native L1 Python module, a compiled FABRIC copy kernel, and reachable
L1 gRPC endpoints. Separate networked machines without this fabric connectivity
are insufficient. No third coordination machine is required.

Build/install this MORI checkout in the normal supported ROCm build environment.
The service uses `mori_umbp_l1`/`umbp_l1`; its runtime does not need the standalone
tier server. Install the matching MORI Python package on ATOM workers for the
lightweight `mori.umbp_l1.NativeL1Memory` client helper. Allocation also needs the
regular MORI IO allocator library and PyTorch MemPool support. Building the L1
extension alone does not install the Python helper or allocator.

For a configured ROCm build, enable `BUILD_IO`, `BUILD_UMBP` and
`BUILD_UMBP_L1_PYTHON`, and explicitly set `GPU_TARGETS` (`gfx1250` on 455), then build:

```bash
cmake --build /path/to/build --target mori_umbp_l1 mori_fabric_copy_kernel
```

The explicit kernel target produces `/path/to/build/lib/fabric_copy.hsaco`.

The separate `mori_umbp_l1` extension must also be on the worker/service Python
path. With CMake's default output directories it is under `build/src/umbp`:

```bash
export PYTHONPATH="/path/to/build/src/umbp${PYTHONPATH:+:$PYTHONPATH}"
```

Use the actual extension output directory if the build overrides it. The new
`mori.umbp_l1` Python package is included by the normal MORI package discovery;
it does not import `mori.umbp` or load the general tier stack.

Start one service per tray (replace peer addresses and the kernel path):

```bash
# Tray A
python tools/umbp_l1_fabric_server.py --listen 0.0.0.0:29620 \
  --peer TRAY_B_IP:29620 --kernel /path/to/fabric_copy.hsaco

# Tray B
python tools/umbp_l1_fabric_server.py --listen 0.0.0.0:29620 \
  --peer TRAY_A_IP:29620 --kernel /path/to/fabric_copy.hsaco
```

This control service uses existing insecure gRPC, so its endpoints are intended
for the trusted cluster network. FABRIC mode requires the kernel; it never falls
back to HIP memcpy or staging. Device ordinals and visibility must match between
each ATOM worker and its local L1 service.

The companion ATOM cross-tray adapter is based on
[PR #3482](https://github.com/ROCm/ATOM/pull/3482) at
`db27ec2198711f2d77dafff8ed0d5621af59d9fb` (open on 2026-11-23).
Install this MORI Python helper and native extension together before enabling
the adapter; #1045 alone does not provide its FABRIC APIs. Configure `umbp_pool` with
`transport: "fabric"`, `address: "127.0.0.1:29620"`, a unique `node_id` per tray,
and a common explicit `group`. Each engine on a tray needs a distinct namespace.
The group must identify identical model revision, KV/hash conventions,
block size, layer/shard layout and speculative configuration; equal region sizes
alone do not establish semantic compatibility. Existing IPC mode stays the default.

UMBP's `NativeL1Memory` helper runs inside the ATOM worker. It allocates the final
KV arena with `make_fabric_mem_pool`, initializes it with an in-place GPU kernel,
and manages descriptor export, Attach/BindLayout, and safe Detach. The helper
owns the tensor and allocator references; it does not create a second arena or
stage KV bytes. Other allocations keep their usual allocator. Imports of Torch,
the native extension and the IO allocator are deferred until the relevant API
is used. The native module exposes `NATIVE_L1_FABRIC_VERSION=1` for compatibility
checks.

ATOM's existing memory provider delegates to these client-side operations:

```python
from mori.umbp_l1 import NativeL1Memory

memory = NativeL1Memory(client, pool_id, group)
arena = memory.allocate(nbytes, device)  # the final KV tensor
memory.bind_layout(arena, num_slots, regions)  # byte offset/unit size pairs
# Use the arena; publish/commit through the existing L1 client.
# After stopping local admission/writes, confirm peer drain before releasing views:
memory.close()
```

Keep `memory` alive while the tensor or derived views are in use. The ATOM
connector still supplies KV layout and owner write progress and prevents slot
reuse when Fetch completion is unknown. FABRIC handles and allocator lifetime
are encapsulated by UMBP; model and attention code use the existing provider
interface.

## Lifetime and failure behavior

Probe is a hint; AcquireRead atomically resolves and pins current source slots.
Local owner Commit timestamps establish readiness. Destination writer counts
prevent overlapping Fetch and publication; source pins prevent eviction/reuse.
ReleaseRead only permits unpinning after confirmed copy completion or confirmed
absence of submitted work. Owner incarnation and operation IDs fence stale and
duplicated control messages, including a delayed Acquire after Release.

Unknown completion keeps mappings, source pins and destination slots, and
quarantines the destination pool. The client exposes `FetchInDoubtError`; ATOM
treats any FABRIC Fetch exception conservatively and stops serving while retaining
its allocation. A timeout never authorizes reuse or fallback into affected slots.
Detach requires acknowledgement from the same service incarnation as Attach.
Lifecycle RPCs wait at most two seconds for a busy pool, then return an error
without releasing memory. Stop admission and drain peers before shutdown.
The UMBP helper retains the tensor and MemPool before the first Attach RPC; a
lost Attach reply cannot drop that retention. An unconfirmed `close()` raises
`L1MemoryInDoubtError` and retains the allocation. Callers must propagate this
failure instead of continuing normal memory teardown.

This version has no automatic recovery, release retry, or receipt garbage
collection. Each service allows 65,536 operation receipts/tombstones per lifetime;
exhaustion rejects new grants until a coordinated drain/restart. A lost Release
may leave source slots pinned. Forced owner/service termination requires
coordinated peer teardown; retaining Python references cannot preserve memory
after process exit. Source `served_out_slots` counts local copies only; completed
remote copies are logged as `[L1Peer] fabric completed` and counted at destination.

The upstream `UMBP_L1_KEEP_COPIES` policy is preserved. Remote read pins and
destination writers also protect slots in its kept list. Its copy count remains
local to a `PoolManager`; it does not guarantee a global number of copies across trays.

## Validation status

The earlier implementation (MORI base `b946dd65`, ATOM base `bd897325f`) passed
two-tray validation on g13-12 and g13-10 on 2026-11-22: all 32 directed GPU paths,
real GPU Event/Commit ordering, slot reuse, and a ten-minute bidirectional soak.
DeepSeek-V4.1-Flash Fetch-off/on comparisons matched inputs, output text and token
counts for 64 requests per arm including warmup; the 16 measured remote-hit
requests each confirmed actual KV copying. Attention used DP4/TP1 per node and
MoE EP4; this was not cross-node tensor parallelism.

The 2026-11-23 dependency alignment rebuilt the native extension and gfx1250
kernel. CPU regressions passed: 56 C++ pool/peer/RPC tests, the copy planner,
12 Python helper tests, and 128 companion ATOM tests (one intentional skip).
All 18 peer tests also passed with `UMBP_L1_KEEP_COPIES=1`, including a new
kept-list/read-pin regression. CPU import-order and ELF symbol-isolation checks
passed. GPU import-order tests were skipped in the CPU-only environment.

The updated PR heads have not been retested with GPUs or model inference. Earlier
results cover bounded workloads and coordinated shutdown, not full model quality,
saturated throughput, crash recovery, or topologies beyond the two tested trays.
