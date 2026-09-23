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

With no NUMA node configured (or `-1`), the tier keeps the original single-buffer
layout and allocation order. A single node retains `MPOL_BIND`. Multiple nodes
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

The local write path derives one advisory NUMA hint per ranged batch. CPU
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

Multi-node tiers prefault buffers in parallel with a total budget of at most
16 workers. Large buffers can subdivide that budget into page-aligned chunks;
small chunks stay serial. Prefault workers use the target node's allowed CPUs
and restore their previous affinity. `UMBP_DRAM_PREFAULT=0` disables prefault.
GPU registration and the transfer path are unchanged.

This policy favors locality over the opportunity to put an entire object in a
remote contiguous run. Its net benefit depends on fragmentation and workload.
Placement alone does not migrate existing objects or replicate shared keys.
Shared-key replication is a separate opt-in mode described below.

## Shared-key replication

For a workload whose GPUs on both sockets read the same logical keys, opt in to
one copy per NUMA node:

```sh
export UMBP_DRAM_NUMA_NODE=0,1
export UMBP_KV_REPLICATION=numa  # default: none
```

This mode requires a masterless DRAM pool. Every configured backend must have
two buffers and the same ordered pair of distinct NUMA nodes. Unsupported
configurations and unknown replication modes fail initialization. The variable
is read by the server's PoolClient, including an automatically started standalone
server; clients continue sending the original keys. Direct C++ users can set
`PoolClientConfig::numa_replication` (the environment, if set, takes precedence).

Both physical keys are checked and reserved in one pool operation. An existing
copy makes the logical Put a no-op; it neither overwrites that copy nor allocates
the missing one. A concurrent reservation causes a retryable failure. New data
moves from the caller to its preferred replica once, then between the two
pending host allocations, before committing. This preserves one D2H transfer
and the existing pending-slot/Clear lifetime rules. A completed copy can survive
failure of its sibling. Whole and ranged puts use the same batched path.

Get prefers the destination GPU's replica and falls back on a miss. CPU and
unknown destinations prefer the first configured node. Exists checks either
copy without renewing read leases. No background read promotion is performed;
a surviving replica remains authoritative after its sibling is evicted.
Internal suffixes are added to every key, including keys already ending in a
NUMA-looking suffix. Enable or disable the mode with a fresh pool.

Replication uses up to twice the storage for the same logical working set.
It is intended for shared keys such as rank-replicated MLA/DSA KV, not keys whose
contents are sharded per rank. The extra host copy and reduced cache capacity
can offset the read-locality gain. Confirm the net benefit on the intended
workload before enabling it in production; microbenchmark bandwidth alone does
not establish serving throughput or latency improvements.

For a short copy-cost diagnostic, enable the existing trace:

```sh
export UMBP_RANGED_CALL_DEBUG=1
export MORI_UMBP_LOG_LEVEL=info
```

`[NumaReplication][dbg]` reports one line per batch that reaches copying:
`copy_bytes_submitted` is the host-to-host payload submitted to the transfer
layer, `copy_us` covers that call's planning, submission, and completion,
`committed_objects` counts newly committed logical objects, and
`degraded_objects` counts those for which only one replica committed. Sum the
per-batch values for totals. Submitted bytes do not guarantee all bytes landed
if the transfer failed; later eviction is not counted as a degraded commit.
Deduplicated/no-copy batches contribute zero and do not emit this line.

These diagnostics work without a master. They add no copy timing calls or log
lines when the trace is off. Client Put traffic remains counted once; internal
replica traffic is a separate quantity. Copy time helps attribute write cost,
but queue growth and serving backpressure must still be checked at the caller.
