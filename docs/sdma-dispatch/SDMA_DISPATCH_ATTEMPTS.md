# Attempts To Use SDMA In The MoRI Intra-Node Dispatch Kernel

This document summarizes the engineering path we took while experimenting with
SDMA in MoRI's intra-node expert-parallel dispatch kernel. It is meant to let
another agent continue the work without re-discovering the same constraints.

The main kernel file is:

```text
src/ops/dispatch_combine/intranode.hpp
```

The launch path is:

```text
python/mori/ops/dispatch_combine.py
```

The SDMA queue device helpers are:

```text
include/mori/application/transport/sdma/anvil_device.hpp
include/mori/application/transport/sdma/sdma_pkt_struct.h
```

## High-Level Goal

The normal intra-node dispatch kernel copies token payloads directly from GPU
threads. We explored a separate SDMA dispatch kernel body:

```cpp
EpDispatchIntraNodeSdmaKernel_body
```

The old normal kernel body was intentionally left alone:

```cpp
EpDispatchIntraNodeKernel_body
```

The goal was to offload the large token payload copy to SDMA while metadata
work remains on GPU warps:

- route each token/expert assignment to a destination PE,
- deduplicate repeated destination PEs per token,
- allocate destination token IDs,
- write metadata, weights, indices, and scales,
- submit token payload copies through SDMA,
- notify receivers only after the copied payload is visible.

## Baseline Behavior

The original `EpDispatchIntraNodeKernel_body` assigns one warp to a token/expert
entry. That warp performs both metadata updates and token payload copy. This is
simple and has good small-token performance, but the token payload copy consumes
GPU execution resources.

The first SDMA variants tried to mimic the normal path closely, while replacing
the payload copy with SDMA queue packets.

## Early SDMA Kernel Separation

The first important architectural decision was to keep a separate SDMA kernel:

```cpp
EpDispatchIntraNodeSdmaKernel_body
```

instead of modifying `EpDispatchIntraNodeKernel_body`. This made it possible to
select SDMA through `MORI_ENABLE_SDMA=1` while retaining the direct-copy path for
comparison and fallback.

The Python launch path was updated so that intra-node dispatch chooses:

```text
EpDispatchIntraNodeSdmaKernel_*  when SDMA is enabled
EpDispatchIntraNodeKernel_*      otherwise
```

## Initial Per-Wave SDMA Submission

The initial implementation had many waves/warps submit SDMA packets. Each call
to the SDMA helper performed queue reservation, packet placement, and queue
doorbell submission.

The important helper became:

```cpp
EpDispatchIntraNodeSdmaSubmitMappedWave(...)
```

It performs:

1. choose an SDMA queue handle,
2. reserve queue space,
3. place one copy packet for each selected lane,
4. submit the queue write pointer once for the batch.

The helper was later force-inlined. We also added profiling spans around:

```text
dispatch_sdma_submit
dispatch_sdma_reserve
dispatch_sdma_place_packet
dispatch_sdma_submit_packet
```

### Finding: Queue Submission Was Expensive

For the copy path, the expensive operation was not just packet placement. The
doorbell/queue submit path inside `handle.submitPacket()` was significant.

With profiling enabled, one `dispatch_sdma_submit` call was around several
microseconds, and a copy warp could perform several such calls per dispatch.

## Cached Hardware Read Index

`SdmaQueueDeviceHandle::ReserveQueueSpace` gained an optional:

```cpp
uint64_t* cachedHwReadIndex
```

The SDMA copy warp keeps a local cache:

```cpp
uint64_t cachedHwReadIndex = 0;
```

and passes it through both copy and completion submission helpers. This reduced
the cost of repeated hardware read-index polling during queue reservation.

The cache is local to the submitting thread/warp path. It should not be shared
between independent queues or blocks.

## PE-Affine Queue Design

We next constrained each block's SDMA queue to a destination PE:

```cpp
int queuePe = blockIdx.x % npes;
if (destPe != queuePe) continue;
```

This made completion accounting simple: the SDMA warp for queue `queuePe`
submitted copies only for `destPe == queuePe`, so it could emit one completion
count for that PE.

### Work Partitioning Challenge

With `destPe != queuePe` filtering, global work partitioning is incorrect:

```cpp
for (int i = globalMetadataWarpId; i < totalAssignments; i += globalMetadataWarpNum)
```

If the only warp that owns assignment `i` belongs to a different `queuePe`, it
will skip the entry and no one else will process it.

The fix was to partition work per queue-PE group:

```cpp
int queuePe = blockIdx.x % npes;
int blockIdxForQueuePe = blockIdx.x / npes;
int blocksForQueuePe = (gridDim.x + npes - 1 - queuePe) / npes;

int globalMetadataWarpId =
    blockIdxForQueuePe * metadataWarpNumPerBlock + metadataWarpId;
int globalMetadataWarpNum =
    blocksForQueuePe * metadataWarpNumPerBlock;
```

Each PE group scans the full assignment list and keeps only entries for its PE.

This preserves correctness, but repeats the metadata scan across PE groups.

## Role-Split Streaming Ring

We then split warp roles within a block:

```text
warp 0:       SDMA copy/consumer warp
warps 1..N:  metadata producer warps
```

Metadata warps do the same work as the normal kernel except the payload copy.
They enqueue copy tasks to a per-producer shared-memory ring:

```cpp
EpDispatchIntraNodeSdmaCopyTask {
  index_t srcTokId;
  index_t destTokId;
  int destPe;
}
```

The SDMA copy warp consumes tasks from producer rings, batches selected lanes,
and submits SDMA copy packets.

### Ring Parameters Used

Important constants used during the later experiments:

```cpp
constexpr int kSdmaWarpId = 0;
constexpr int kMaxMetadataWarps = 15;
constexpr int kSlotsPerProducer = 32;
constexpr int kSlotsPerSdmaBatchPerProducer = 4;
constexpr int kProducerLaneGroupSize = 16;
```

With `dispatch_warp_per_block=16`, this means:

```text
1 SDMA warp
15 metadata warps
up to 4 consumed slots per producer per SDMA iteration
theoretical max selected tasks per submit batch = 15 * 4 = 60
```

In practice, for `max_tokens=128`, the average tasks per submit was much lower
because the workload per block/queue was small.

## Copy-Warp vs Metadata-Warp Profiling

We added role-specific spans:

```text
dispatch_sdma_metadata_warp
dispatch_sdma_copy_warp
```

Latest representative profile before several completion experiments:

```text
metadata warp median: ~17 us
copy warp median:     ~45-50 us
```

This initially suggested the copy warp was the bottleneck. However, this needed
careful interpretation because there are many metadata warps and one copy warp
per block.

We added more spans and counters:

```text
dispatch_sdma_consumer_poll
dispatch_sdma_consumer_commit
dispatch_sdma_completion
dispatch_sdma_submit_call
dispatch_sdma_empty_poll
dispatch_sdma_non_empty_poll
dispatch_sdma_active_count_bit0..5
```

The active-count bit instants let us reconstruct:

```text
sum(activeCount)
avg tasks per submit = sum(activeCount) / submit_call_count
```

For one focused profile at `max_tokens=128, blocks=32, wpb=16`, we observed:

```text
submit_calls:          2516
active_tasks_sum:      16116
avg_tasks_per_submit:  ~6.4
empty_poll_fraction:   ~59%
```

This meant the SDMA warp was not usually submitting full batches; for small
token counts, batches are naturally small.

## Removing PE Affinity

We then removed:

```cpp
if (destPe != queuePe) continue;
```

The queue selected by `queuePe = blockIdx.x % npes` could submit copies to any
destination PE. This avoids repeated PE-group metadata scans and improves task
batching.

The metadata work partition becomes global again:

```cpp
int globalMetadataWarpId = blockIdx.x * metadataWarpNumPerBlock + metadataWarpId;
int globalMetadataWarpNum = gridDim.x * metadataWarpNumPerBlock;
```

### New Synchronization Problem

Once a single queue can submit copies to many destination PEs, one total
`submittedCount` is not enough. Receivers wait per destination PE/source PE. We
therefore need per-destination completion accounting:

```cpp
submittedCountByPe[destPe]
```

The first no-affinity versions submitted completion atomics per destination PE.

## Completion Protocol Variants

### 1. Per-Block Remote Completion Fanout

Each block's SDMA warp submitted completion atomics for the destination PEs it
touched. This means many blocks can contribute to the same destination signal.

This improved copy batching but made phase-2 receiver waits expensive.

Representative profile:

```text
dispatch_sdma_copy_warp:       ~74.7 us
dispatch_sdma_completion:      ~40.8 us summed per copy warp
dispatch_wait_peer_token:      ~70 us on global warp 0
```

The receiver had to wait for all completion atomics from many blocks/queues.

### 2. Batched Completion Per Block

We changed completion submission so each block reserves queue space once, places
all nonzero destination completion atomic packets, and submits once:

```text
TIMESTAMP start
ATOMIC_ADD completion destPe 0, if nonzero
...
ATOMIC_ADD completion destPe 7, if nonzero
TIMESTAMP end
submit once
```

This reduced GPU-side completion submission overhead but did not eliminate the
receiver-side wait cost when many blocks still emit completion atomics.

Representative improvement:

```text
dispatch_sdma_completion: ~40.8 us -> ~7.9 us
dispatch_sdma_copy_warp:  ~74.7 us -> ~39.2 us
```

However, phase 2 was still elevated because the receiver still waited for many
remote atomic packets to arrive.

### 3. Post-Grid-Barrier Per-Queue Completion Aggregation

We then aggregated by `queuePe`: after all blocks had submitted copy packets,
one owner block per `queuePe` submitted the batched completion for that queue.

This required a temporary matrix:

```text
sdmaCompletionCounter[queuePe][destPe]
```

The SDMA copy warps accumulated their local per-destination counts into this
matrix. After a grid barrier, the owner block for each `queuePe` submitted the
completion batch.

This significantly reduced phase-2 wait:

```text
dispatch_wait_peer_token global-warp median:
  before: ~70 us
  after:  ~30 us
```

Focused non-profiler `max_tokens=128, blocks=32` result:

```text
dispatch: ~84-88 us
e2e:      ~140-141 us
```

### 4. Local Queue Completion

The next idea was to make SDMA completion local. Instead of using remote
completion signals per destination PE, each queue emits a local queue-drain
completion after all copy packets are submitted. The sender waits for these
local completions before publishing `numTokenSignal`. The receiver only waits on
`numTokenSignal`.

Ordering:

```text
all blocks submit SDMA copy packets
grid barrier
one local completion packet per queuePe, ordered behind that queue's copies
global warp waits local queue completion signals
publish numTokenSignal to peers
receiver sees numTokenSignal only after copies are complete
```

This is the cleanest protocol so far conceptually. It turns `numTokenSignal`
into the remote-visible completion notification and avoids per-destination SDMA
completion atomics.

Focused non-profiler `max_tokens=128, blocks=32` result:

```text
dispatch: ~87-88 us
e2e:      ~138 us
```

The full sweep for this local-completion variant is in:

```text
bench_results/sdma_local_completion_blocks_tokens_20260615/summary.csv
```

## SDMA Timestamp Packet Instrumentation

We added:

```cpp
SDMA_SUBOP_TIMESTAMP_GET_GLOBAL = 2
anvil::CreateTimestampPacket(...)
```

The timestamp packet address encoding must preserve low address bits in their
documented positions:

```cpp
packet.ADDR_LO_UNION.addr_31_0 = static_cast<uint32_t>(addr & ~uintptr_t{0x7});
packet.ADDR_HI_UNION.addr_63_32 = static_cast<uint32_t>(addr >> 32);
```

An earlier attempt used `addr >> 3` and caused the SDMA engine to write to the
wrong address, producing GPU memory faults.

With correct encoding, timestamp packets can bracket completion atomic batches.
For the return-atomic batched completion path, the measured timestamp delta was
roughly:

```text
~16.5 to ~16.8 timestamp-converted units
```

Treat absolute units carefully until the SDMA timestamp clock is calibrated
against `wall_clock64()`. Relative comparisons are still useful.

## No-Return Atomic Experiment

The current workspace is detached at:

```text
f56a005e0776c10113986a76614c3a6d8def0677
```

with a local patch adding:

```cpp
const unsigned int SDMA_ATOMIC_ADD64_NO_RETURN = 111;
anvil::CreateAtomicAddNoReturnPacket(...)
```

and using this helper for completion atomics.

The hope was that no-return atomics would reduce completion batch overhead
because the SDMA engine does not need to return the atomic result.

Focused profile result:

```text
trace_intranode_rank*_0616_215652.json
```

Compared to the prior return-atomic measurement, there was no clear reduction:

```text
SDMA completion timestamp delta median: ~16.88
dispatch_sdma_completion median:        ~9.12 us
dispatch_sdma_copy_warp median:         ~32.24 us
dispatch_wait_peer_token global warp:   ~30.20 us
```

Conclusion: operation `111` did not obviously reduce completion atomic batch
cost in this setup.

## Important Commits

Known relevant commits in local history:

```text
7704eaee use local sdma completion
f56a005e no q-affinity, need 8 atomic-add per queue/block
0dc78c59 more efficient task sharing
25c1523a add cachedHwReadIndex
739c1395 wave roles
676e5ec8 v1
```

Current detached state for latest experiment:

```text
HEAD: f56a005e
local diff: no-return atomic operation 111 for completion packets
```

## Key Lessons

1. Keep the normal direct-copy kernel untouched.

   The SDMA path has enough protocol complexity that it should remain separate
   until it clearly wins.

2. SDMA queue submission cost matters.

   `handle.submitPacket()` and queue reservation/fencing are significant fixed
   costs. Batching helps, especially for small token counts.

3. Completion protocol dominates correctness and latency.

   Copy submission is not enough; receivers must not consume data before all
   relevant SDMA copies are complete.

4. Remote completion atomics are expensive.

   Per-block/per-destination completion atomics raised phase-2 wait time
   substantially. Batching reduced GPU-side overhead, but remote atomic
   execution itself remained costly.

5. Local queue completion is conceptually better.

   It waits locally for queue drain before publishing `numTokenSignal`, reducing
   remote completion atomics. This is the most promising direction so far.

6. Queue/destination affinity is a tradeoff.

   PE-affine queues make completion simple and may have better locality, but
   require repeated PE-group scans. Non-affine queues batch better and avoid
   repeated scanning, but completion accounting is harder.

7. Profiling changes behavior.

   Profiler spans and instants add overhead, especially when nested in tight
   loops. Use profiler-disabled sweeps for headline performance and profiler
   traces only to explain bottlenecks.

## Open Questions / Next Work

1. Revisit local queue completion as the primary direction.

   The latest no-return atomic experiment was run from `f56a`, not the local
   completion commit. If continuing performance work, compare against:

   ```text
   7704eaee use local sdma completion
   ```

2. Calibrate SDMA timestamp units.

   The SDMA timestamp packet works, but the absolute unit should be calibrated
   against a known delay or GPU wall clock before reporting absolute time.

3. Reduce metadata and copy-warp overhead without increasing completion cost.

   For small token counts, average tasks per submit can be low. For large token
   counts, queue throughput and SDMA engine behavior dominate.

4. Consider alternative completion notifications.

   If `numTokenSignal` is always published only after local SDMA queue drain,
   remote SDMA completion signals may not be needed at all for dispatch.

5. Re-check queue affinity.

   Non-affine queues simplify work partitioning and improve task batching, but
   may be less optimal for SDMA routing. A hybrid model might be necessary.

