# Step 2: design checklist

Write one design note per candidate ([../templates/design-note.md](../templates/design-note.md))
before any kernel code. Read the operator READMEs first
(`python/mori/ops/<op>/README.md`): they hold measured results and dead ends.

## 1. Dependency structure

- Direction: compute to communication (GEMM + AR or RS, MoE combine), or
  communication to compute (AG + GEMM, MoE dispatch).
- The unit that can be sent or consumed: how tiles map to chunks.
- Where the consumer runs, in the next kernel or the same kernel. This decides who
  issues the transfer: the block that completes a chunk ("last finisher"), or a
  dedicated communication workgroup.

## 2. Layout and chunking

- Pick the shard axis that makes each destination's data contiguous; the copy engine
  cannot gather.
- Write straight into the peer's receive slot and avoid staging copies.
- Choose `n` from the step-1 model and keep puts above the knee (`G950-SDMA-04`);
  consider a smaller last chunk.
- Produce tiles chunk by chunk, in send order, and rotate the destination order by rank
  so that the ranks do not all hit the same peer first.

## 3. Transport

- SDMA takes no CUs but costs a few µs per put (`G950-SDMA-01`, `G950-SDMA-02`).
- LSA (CU loads and stores over xGMI) has lower latency but runs on the GEMM's own
  waves; on Flash fused-LSA lost everywhere (`G950-GAR-04`). Below ~256 KB per put, LSA
  is expected to win (`G950-SDMA-04`).
- One queue per peer link: one peer is one link, and splitting a peer's run across
  queues only pays the per-packet cost again (`G950-SDMA-05`).
- Issue to different peers from different lanes, not back to back from one thread
  (`G950-SDMA-02`).

## 4. Synchronization protocol

Write every step down; this is where the bugs are.

- **Producer:** after its stores retire (`s_waitcnt vmcnt(0)` and a barrier), one thread
  per block does one `acq_rel` atomic add on the chunk counter. That atomic is the
  release (`G950-SYNC-01`); an added fence costs hundreds of µs (`G950-SYNC-02`).
- **Issue:** who posts which put, and when.
- **Consumer:** one acquire after the flag; none per tile or per poll (`G950-SYNC-03`).
- **Completion:** a drain kernel after the GEMM (quiet, then flags), not a quiet inside
  it (`G950-SYNC-04`).
- **Buffer reuse:** a barrier at the end of call e-1 proves every peer wrote, not that
  it read. When the output is the receive buffer, post credits and wait for peer d's
  credit before the first put to d.
- **Epochs and counters:** kept on the device, advanced by the kernels, compared
  wrap-safely, so a graph replay behaves like a fresh launch. Reset a counter on its
  last arrival (`G950-CTR-01`); a modulo election on a running counter breaks at wrap
  (`G950-CTR-02`).

## 5. Resources and numerics

- Registers, LDS and occupancy after the communication code is added: no spills, and the
  K-loop schedule unchanged (`G950-ISA-01`).
- CUs held by a communication workgroup; a grid barrier needs every block resident at
  once.
- Window offsets in a buffer resource are 32-bit; windows of 4 GiB or more need
  `per_rank_vmm`.
- Reduction order and wire precision, and the correctness rule they imply.

## 6. Prediction and ranking

- Per candidate: the predicted time per bucket, the predicted timeline (when the first
  put starts, how long the tail is), the risks, and the measurement that would show the
  design cannot reach its prediction.
- Rank by expected gain and risk, and build first the simplest candidate that can reach
  `T_real`. That is often stream-level chunked overlap: the GEMM split into chunks and
  overlapped with chunked transfers on a second stream, with no kernel changes. Keep it
  as a baseline even if a fused kernel replaces it.

## 7. A design that can be taken apart

- Every mechanism behind a compile-time switch (`const_expr`); every switch, and any
  configuration that is not a closure scalar, in the kernel name; a separate
  `FLYDSL_RUNTIME_CACHE_DIR` per variant. A stale cache shows up as wrong results or
  hangs.
- At least these variants runnable: the GEMM with communication off; signals only, no
  data; data without waiting; the full fused kernel.
- Timeline stamps compiled in from the start, off by default.
- New behaviour off by default, behind its switch, until it is accepted.
