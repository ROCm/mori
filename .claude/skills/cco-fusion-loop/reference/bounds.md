# Step 1: baseline and bounds

How much can fusion give for this shape, and is it worth building?

## Notation

| Symbol | Meaning |
|---|---|
| bucket | a range of M served by one configuration |
| `T_split` | the unfused path: the fastest GEMM, then the fastest collective, captured in one graph |
| `a` | overlappable compute: the GEMM plus whatever the fused GEMM takes on (epilogue stores, put submission, quantization) |
| `b` | overlappable communication: the transfer legs that depend only on this GEMM's output, as timed in the split path, so it already includes one put's fixed cost per destination |
| `T_serial` | the legs that must wait for something else, such as the reduce and the all-gather of an all-reduce, which wait for every rank's scatter |
| `n` | chunks (puts) per destination |
| `t0` | fixed cost of one put; 4-9 µs on SDMA depending on how it is issued (`G950-SDMA-01`), about 2 µs per extra packet from one thread (`G950-SDMA-02`) |

## Measure

In graph mode, one shape per process, maximum over ranks
([measurement-protocol.md](measurement-protocol.md)):

1. The fastest standalone GEMM.
2. The fastest collective for the message size.
3. The split path end to end, its per-phase times from a kernel trace, and the
   end-to-end time again with tracing off. Use the phase times measured inside the
   pipeline: the preceding work and the cache state make them differ from standalone
   times (`G950-GAR-01`, caveat).
4. Contention: the GEMM and the collective launched concurrently on two streams, on
   independent buffers. How much each slows the other gives `a` and `b` under
   contention, before any fused code exists.

For gemm_ar, `bench_gemm_ar.py --mode gemm-only | split-sdma | split-lsa | split-rccl`
and `sweep.py` cover items 1 to 3. Item 4 has no tool yet.

## Ideal bound

Split the communication by dependency: what can run while this GEMM runs (`b`), and
what cannot (`T_serial`).

```text
T_ideal = T_serial + max(a, b)
```

`max(compute, communication)` is the special case `T_serial = 0`, where every
communication leg can overlap the GEMM. A GEMM + all-reduce that fuses only the
reduce-scatter transfer has the reduce and the all-gather in `T_serial`.

## Realistic bound

Overlap is a pipeline. With `n` chunks per destination:

```text
compute-bound:        T_real ≈ T_serial + a + b/n + t0          # the last chunk's transfer is exposed
communication-bound:  T_real ≈ T_serial + a/n + b + (n-1)*t0    # the first chunk's compute is exposed,
                                                                # and the per-put costs add up
```

More chunks shorten the exposed part but add fixed costs, and small puts lose
bandwidth (`G950-SDMA-04`), so `n` has an optimum: sweep it. A GEMM with fewer tiles
than CUs finishes all its tiles at about the same time and cannot be pipelined at all;
that is the reason behind gemm_ag's fusion threshold (`G950-ENV-02`).

The formulas assume that chunks complete at an even rate, that each destination has its
own queue and destinations proceed in parallel, and that `a` and `b` are already
adjusted for contention.

## Other levers

Next to the fusion bound, estimate what the other levers are worth on the split path:
a narrower wire (FP8), another collective algorithm or transport, a faster GEMM. They
interact: a shorter collective raises the fusion ceiling, and a faster GEMM lowers it.
On DeepSeek-V4.1-Flash the GEMM and the wire were each worth more than the overlap
(`G950-GAR-02`, `G950-GAR-05`).

## Worked example

gemm_ar on DeepSeek-V4.1-Flash `wo_b`, TP4, M=16384 (`G950-GAR-01`), in µs:

| Phase | split-SDMA | fused-SDMA |
|---|---:|---:|
| GEMM (fused: plus scatter submission) | 149.9 | 202.2 |
| Scatter transfer (fused: the remaining drain) | 728.0 | 587.3 |
| Reduce | 32.4 | 32.5 |
| Gather | 727.8 | 727.7 |
| Whole call, tracing off | 1638.8 | 1549.6 |

Fusion can only change the first two rows: 877.9 µs split, 789.5 µs fused, 88.4 µs
saved.

- **Ideal:** `max(149.9, 728.0) = 728.0`, so at most 149.9 µs can be saved;
  `η_ideal` = 59%.
- **Realistic** (communication-bound, `n` = 8, `t0` = 4-9 µs): with `a` = 149.9, which is
  all that is known at step 1, `18.7 + 728.0 + 7*t0` = 775-810 µs. With the measured
  fused `a` = 202.2, 781-816 µs. Measured: 789.5 µs. This comparison is inferred; the
  README reports only the measurements.
- **Reading:** the design is at, or close to, its realistic bound, so more tuning of this
  kernel has little to give. The next gain needs fewer bytes or a different structure,
  and the FP8 wire delivers it (`G950-GAR-02`).

Across M on this model, fusing collects about half of the ideal ceiling
(`G950-GAR-03`).

## Go or no-go

Do not fuse when the realistic saving is below the noise band or below the contract's
minimum, or when the GEMM has fewer tiles than CUs. Record the result with its numbers
and move on to the other levers.

Fusion can also remove work: an intermediate write and read of HBM, staging copies,
launches. A fused result can therefore beat a bound derived from the split path. Treat
that as a finding to explain (check correctness first, then the measurement, then the
model), not as a win.

## bounds.md

Per bucket: the measured components with their run IDs, `T_split`, `T_ideal`, `T_real`
for each `n` in the sweep, the other levers and their estimates, and the go or no-go
decision with its reason.
