# Shared FSDP SDMA outputs

`MoriSdmaAllGatherPool` bounds persistent output memory while retaining
parameter-contiguous zero-copy and stable registered addresses. Keep a separate
`MoriSdmaAllGather` instance per FSDP parameter group; share only its pool.

```python
from mori.ccl.torch_fsdp import MoriSdmaAllGather, MoriSdmaAllGatherPool

pool = MoriSdmaAllGatherPool(
    [largest_block_bytes, largest_block_bytes, root_bytes],
    group=process_group,
    device=device,
)
for index, block in enumerate(model.layers):
    block.set_custom_all_gather(
        MoriSdmaAllGather(output_pool=pool, buffer_index=index % 2)
    )
model.set_custom_all_gather(MoriSdmaAllGather(output_pool=pool, buffer_index=2))
pool.initialize()
```

This assignment is for sequential blocks with at most one prefetched block.
The non-resharding root needs its own slot. This is not a general schedule
planner: deeper prefetch, branches, repeated modules, or multiple simultaneously
unsharded groups may need more slots. A live lease conflict raises an error.
Install the backends before the first unshard. Switching an already-used group
to a new pool may require copying into its existing parameter storage to
preserve saved aliases, rather than adopting the new arena's views.

For an ordinary homogeneous-dtype parameter group, its required capacity is
the sum of its padded shard element counts, multiplied by the shard world size
and dtype element size. Capacities must be positive multiples of four bytes
and identical across ranks. Slot assignments must also be identical across
ranks: SDMA applies the local arena offset to each peer's arena. Slots never
grow or move. Construct the pool and perform the same gather/release schedule
on every rank.

Call `initialize()` collectively after constructing every adapter and before
training or any subgroup call. It validates ordered group members, capacities,
and the ordered adapter/slot/mode assignments before raw SDMA can run. Optional
`group_key` strings identify adapters; without keys their construction order is
their identity. Bindings are frozen after initialization. All adapters sharing
a pool must use the same `zero_copy_output`. Dynamic full-pool collective order
must still match on every rank; static validation does not prove that schedule.

Only the exact process group bound to the pool may use its SDMA collective.
Valid strict subgroups use independent Native staging, with no pool-wide
readiness operation. If existing parameters alias a pool slot, its local lease
is retained through copy-out and compute. A rank-major staging-only output may
release its lease after copy-out; an aliased parameter output must not.

On reshard, FSDP calls the backend's `release_output()` notification. The
adapter records consumer completion and returns the group lease. Allocation
waits on that event before any input packing can write the output. Collective
execution also waits on the local event, then performs a one-element SDMA all-gather
into a disjoint control region before writing parameter data. Its generation
flags confirm that every peer has passed its own consumer event. Local events
alone cannot prevent a fast peer overwriting a slow peer's still-live parameters.
The readiness and data operations use the same collective with monotonic
generation tokens; an event orders complete operations when callers change
streams. This readiness operation adds latency and must be included in
performance measurements.

FSDP does not manage MORI reuse events or pool state. The non-pooled adapter
also records its last consumer on release and waits before subsequent packing
and communication. FSDP retains its native post-forward shard synchronization
and the generic parameter identity/version-counter protections.

Release is idempotent. Failed input preparation releases a lease on a stream
ordered after the queued input work. If a collective may have partially started
before raising, the pool is marked failed and cannot be reused or collectively
closed; terminate the distributed workers. This is not a promise to recover an
arbitrary failed distributed step or a hardware error.

The slots are slices of one arena in a dedicated PyTorch memory pool. Only the
arena base is registered: the compatible MORI IPC path does not compensate for
an ordinary caching-allocator suballocation's offset from its allocation base.
Slot offsets are fixed, identical across ranks, and aligned to 16 bytes.

The arena and its single IPC registration survive lease release. The
steady-state arena size is the sum of 16-byte-rounded slot capacities plus a
16-byte-rounded `4 * world_size` control region, rather than the sum
of full outputs of all groups. It is not a guarantee about total training peak:
activations, gradients, optimizer state, allocator reservation, and fallback
copies also contribute. Pool capacity is visible as `pool.allocated_bytes`.

Call `pool.close()` collectively after all groups have resharded and before
SHMEM or process-group teardown. It unregisters the slots; it does not invalidate
outstanding tensor objects or free storage still referenced by parameters.
Automatic garbage collection does not perform collective cleanup.

Requires the matching PyTorch layout interface and `AllGather.release_output`
hook. Supports eager training; CUDA graphs are rejected. Unsupported layouts
use the rank-major copy-out fallback. Integer post-forward reshard gathers use
the native subgroup collective, then restore the same parameter storage. The
fallback is correct but is not zero-copy. The default adapter without an
explicit pool retains the previous per-group allocation behavior.
In particular, non-pooled `zero_copy_output=False` is a legacy resident-buffer
mode, not a memory-saving mode. Use an explicitly shared pool for bounded
registered output capacity; this change does not silently switch it to Native
or introduce per-step IPC registration.
