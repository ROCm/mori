# Design note: <candidate-id>

- Task: <task> (contract revision <rev>)
- Operator and buckets: <op>, M in <buckets>
- Author: <agent or person>, <date>

## 1. Dependency structure
Direction; the unit that is sent or consumed (tiles to chunks); where the consumer runs;
who issues the transfer.

## 2. Layout and chunking
Shard axis; destination layout (contiguous per peer, no staging); `n` per destination
and the put size it implies; tile order; destination rotation.

## 3. Transport
SDMA, LSA or both, and why for these sizes; queues per peer; which lanes issue.

## 4. Synchronization protocol
Producer release; issue; consumer acquire; completion (drain); buffer reuse (credits);
epochs and counters (reset or wrap-safe compare).

## 5. Resources and numerics
Expected VGPR / LDS / occupancy change; CUs held; grid residency; reduction order and
wire precision; the correctness rule this implies.

## 6. Prediction

| Bucket | T_split | T_ideal | T_real | Predicted T_fused | First put at | Tail |
|---|---:|---:|---:|---:|---:|---:|
| | | | | | | |

Facts used (IDs): <...>

Risks: <...>

Falsifier: the measurement that would show this design cannot reach its prediction.

## 7. Switches and variants

| Switch | Default | What it enables |
|---|---|---|
| | | |

Variants that must run: communication off; signals only; data without waiting; full.
