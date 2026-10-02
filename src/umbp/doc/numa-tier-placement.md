# NUMA placement for a DRAM tier

UMBP can split a DRAM tier into one buffer per NUMA node and prefer the writer
GPU's local buffer. All buffers remain in one backend and one shared keyspace.
Readers on any GPU can still access any object.

## Configuration

```sh
export UMBP_DRAM_NUMA_NODE=0,1
# Optional: fail if a mapping cannot be bound, and forbid physical-page spill.
export UMBP_DRAM_NUMA_STRICT=1
# Optional: 1 for serial prefault; 0 (default) selects up to 16 workers.
export UMBP_DRAM_PREFAULT_THREADS=0
```

With no NUMA node configured (unset, blank, or `-1`), the tier keeps the original
single-buffer layout and allocation order. A single node retains `MPOL_BIND`. Multiple nodes
use `MPOL_PREFERRED` unless strict mode is enabled. Duplicate nodes, negative
nodes mixed into a list, and capacities too small for one allocator page per
node are rejected.

Binding policy and syscall failure handling are separate:

| Nodes | STRICT | Policy | If `mbind` fails |
|---|---|---|---|
| Unset / `-1` | Either | No binding | `mbind` is not called |
| One node | `0` (default) | BIND | Warn and continue |
| Multiple nodes | `0` (default) | PREFERRED | Log degradation and continue |
| One or multiple nodes | `1` | BIND | Free the mapping and fail allocation |

Without STRICT, `ENOSYS` emits one unavailable warning and continues. With
STRICT, every binding failure, including `ENOSYS`, is fatal: startup must not
claim strict placement on a system that cannot bind memory. A successful BIND
still restricts physical pages to the selected node even when STRICT is zero;
STRICT controls error handling, not whether that policy can spill on a fault.

With `UMBP_DRAM_USE_HUGEPAGES=1`, reserve huge pages per node, not only in
total. The hugetlb reservation made at `mmap` time is global, but a fault under
BIND takes pages only from the bound node, so a node whose own pool is short
fails the fault even though the global count looked sufficient. Reserve at
least each buffer's size on its node, for example
`echo N > /sys/devices/system/node/node<i>/hugepages/hugepages-2048kB/nr_hugepages`.

The capacity is split evenly in whole allocator pages, with the remainder in
the final buffer. The policy JSON's DRAM `numa_node` also accepts a list:

```json
{"type": "dram", "capacity": "40GiB", "numa_node": [0, 1]}
```

Policy-created DRAM backends inherit the configured hugepage, prefault, strict
binding and prefault thread options. Python exposes `dram.numa_nodes`,
`dram.numa_strict`, and `dram.prefault_threads`. The deprecated `dram.numa_node`
property reads the first node (or `-1`); assigning it replaces the entire list.
The standalone client's child server receives the same configuration.

`UMBPHostMemAllocator.alloc(..., numa_node=...)`, including SGLang's HiCache L2
allocator, keeps its scalar API and existing best-effort `MPOL_BIND` behavior.

## Allocation and startup

The local write path derives one advisory NUMA hint per ranged batch and one per
object for whole-object `Put`/`BatchPut`. CPU
buffers and unknown GPUs have no preference. If the preferred buffer can
satisfy a request, contiguous pages are preferred over scattered pages within
that buffer. Otherwise, a global capacity check precedes allocation of the
local remainder and then pages from other buffers. Insufficient global space
leaves all bitmaps unchanged. HBM and SSD backends ignore the hint.

Logical page spill and physical memory spill are distinct: the former uses
another tier buffer when the local buffer is full; `MPOL_PREFERRED` also lets
the kernel back a buffer with another node's physical pages when necessary.
Startup logs distinguish target placement from sampled physical placement.
Failure to query placement is logged rather than inferred as success.

Every tier of at least 64 MiB prefaults in parallel with a total budget of at
most 16 workers (`UMBP_DRAM_PREFAULT_THREADS=1` restores the serial path).
Large buffers can subdivide that budget into page-aligned chunks; small chunks
stay serial. A bound buffer's workers run on its node's allowed CPUs. An
unbound buffer's workers alternate between nodes in whole rounds, so first
touch splits it evenly across sockets instead of filling one node first.
Workers restore their previous affinity. `UMBP_DRAM_PREFAULT=0` disables
prefault. GPU registration and the transfer path are unchanged.

This policy favors locality over the opportunity to put an entire object in a
remote contiguous run. Its net benefit depends on fragmentation and workload.
It does not migrate existing objects or replicate shared keys across sockets.

## Measured effect

One MI355X node (two sockets, eight GPUs), DeepSeek-V4-Pro at TP8, SGLang with
the UMBP linker and an agentx replay. Split tier (`UMBP_DRAM_NUMA_NODE=0,1`)
against the single-buffer tier:

| Concurrency | DRAM tier | Window | Runs (single / split) | Total token throughput | TTFT p50 |
|---|---|---|---|---|---|
| 16 | 512 GiB, 4K pages | 900 s | 3 / 5 | within 1% | no consistent difference |
| 128 | 740 GB, hugetlb, STRICT | 3600 s | 2 / 2 | +0.2% | -0.4% |

Placement does not change serving throughput at these points. At concurrency 16
tier reads overlap compute. At 128 the load wait that reaches the critical path
is about 1.3% of forward time with or without the split. A hugetlb tier
reserved to capacity already fills node 0 and then node 1.

The split does shorten startup. A standalone server with a 740 GB tier is ready
in 64-81 s with 4K pages and in 10-13 s with hugetlb. A single-buffer tier used
to take 274 s and 36 s: one thread prefaulted it, which placed the whole tier on
that thread's node, and registering a tier packed onto one node for GPU access
takes several times longer. Its workers now alternate between nodes, and it is
ready in 79-90 s and 12-14 s.

Replicating shared keys on both nodes was measured at concurrency 128 and not
adopted. Throughput fell by 10.9%: storing each shared key twice raised
recomputed prefill by 25%, and exposed load wait rose by 30%.
