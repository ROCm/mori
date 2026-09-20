# mori.ops.gemm_a2a — fp8 GEMM fused with an all-to-all

Measurements for `mori.ops.gemm_a2a`, the column-sharded all-to-all fused into
the epilogue of mori's 8-wave fp8 GEMM.

## What is measured

Every rank holds its own `A [M, K]` and a replicated `B [N, K]`, computes the
whole `C = A @ B.T`, and sends column block `j` to rank `j`. Rank `d` ends up
with `[world*M, shard_n]` bf16, source rank `r` at rows `[r*M, (r+1)*M)`.

| mode | what it does |
|---|---|
| `gemm-only` | the GEMM alone, to size the ceiling |
| `split-lsa` | `gemm()`; then a copy kernel reads `[M,N]` and vector-stores into the peers |
| `split-sdma` | `gemm()` into `[dst][M][shard_n]`; then one SDMA push per destination |
| `fused-lsa` | C stored straight into the peers from the epilogue |
| `fused-sdma` | C staged per destination, pushed by the copy engine per chunk |

`split-sdma` and `fused-sdma` use the **same** GEMM with the epilogue's put tail
compiled out, so they differ in one thing only.

## Conditions

8x MI355X (gfx950), TP8, `M=2048 N=18432 K=8192`, `shard_n = 2304`,
`--quant ptpc`, CUDA-graph replay, median of 21 iterations after 8 warmups, max
over ranks. **Three independent launches per configuration**; the table reports
the median of the three and the spread between them.

The shape is gcnasm's `opus_gemm_a2a_lsa` 8-rank full-N configuration, so the
*shapes* are comparable. Its numbers are bf16 and this is fp8, so only the
fused-vs-split **ratio** transfers, never the absolute time.

> Measured on an idle box, verified with `rocm-smi --showpids` beforehand. An
> earlier attempt on a shared box put the same configuration at 635.5us and
> 787.6us on consecutive runs -- 24% apart, wider than anything being measured.
> Those runs were discarded, not averaged in.

## Results

| configuration | median (us) | spread | comm (us) | vs `split-lsa` |
|---|---:|---:|---:|---:|
| `gemm-only` | **325.0** | 1.2% | — | — |
| `split-lsa` | 952.4 | 0.2% | 627.4 | — |
| `split-sdma` | 630.3 | 0.5% | 305.3 | −33.8% |
| `fused-lsa` (rotated, stripe 1) | 642.6 | 2.0% | 317.6 | −32.5% |
| `fused-lsa` (no rotation) | 618.2 | 1.6% | 293.2 | −35.1% |
| `fused-lsa` (rotated, stripe 3) | 623.4 | 0.2% | 298.5 | −34.5% |
| `fused-sdma --chunks 1` | 622.0 | 6.5% | 297.1 | −34.7% |
| `fused-sdma --chunks 2` | 552.5 | 0.4% | 227.5 | −42.0% |
| **`fused-sdma --chunks 4`** | **522.3** | 0.6% | **197.3** | **−45.2%** |
| `fused-sdma --chunks 8` | 532.4 | 0.5% | 207.5 | −44.1% |
| `fused-sdma --chunks 16` | 567.1 | 0.4% | 242.1 | −40.5% |

"comm" is `median - gemm-only`, i.e. everything the GEMM does not account for.
It is the number the fusion is trying to move, and quoting only the end-to-end
figure understates it by the GEMM's share.

**Headline: `fused-sdma --chunks 4` at 522.3us, 45.2% under `split-lsa` and
17.1% under `split-sdma`. Against its own split baseline the comm half falls
305.3 -> 197.3us, −35.4%.**

## Reading the table

**SDMA beats LSA on both halves, and that is the opposite of what gcnasm
reports.** Its README has Direct (fused) LSA ahead of Fused SDMA on all seven
8-rank shapes. Here `fused-sdma` (522.3) beats `fused-lsa` (618.2) by 15.5%,
and even `split-sdma` (630.3) beats the best `fused-lsa`.

The difference is not a contradiction, it is the shape. gcnasm's experiment is
**bf16** and this is **fp8**: same output bytes, half the input bytes, so the
GEMM is roughly twice as fast and the compute/comm ratio is half. LSA wins when
there is enough compute to hide CU-issued stores behind; SDMA wins when the
copy engines' independence from the CUs matters more. Halving the compute moves
the shape across that line. This is the same mechanism recorded for `gemm_ar`,
where `fused-sdma` won at compute/comm 0.22 while gcnasm's 2.8 favoured LSA --
the ratio, not the transport, is what decides.

**Chunking is the whole of the fused-SDMA win.** At `--chunks 1` the fused path
is 622.0us, statistically the same as `split-sdma`'s 630.3 -- one put per
destination cannot start until that destination's last tile is written, which is
near the end of the GEMM either way, so there is nothing to overlap. Chunks 2
and 4 let a destination's earlier rows leave while its later ones are still
being computed, and that is worth 100us. Past 4 it reverses: at 16 chunks each
put is 576 KiB, below the knee in the SDMA bandwidth curve, and the per-packet
cost starts dominating. The curve's minimum at 4 is shallow -- 4 and 8 are
within 2% -- so this is not a knife edge.

**Destination rotation does not pay here, and may cost.** `fused-lsa` is
fastest with rotation *off* (618.2 against 642.6 rotated). gcnasm measured the
rotation worth 17-27% at 8 ranks, but for `M >= 8192`; this is `M = 2048`, a
quarter of its smallest striped shape. With only 16 row tiles there is little
for a rotation to spread. The knob is kept because the effect is shape-dependent
by construction, not because this configuration wants it on.

**The one noisy cell is `fused-sdma --chunks 1`** at 6.5% spread (618.4 / 622.0
/ 658.5). It is also the configuration with the least overlap, so its timing is
most exposed to when the last tile happens to land. Not chased further: it is
not a configuration anyone would ship.

## Correctness

All five modes validate at relL2 **0.0016595761784107884** -- the same value to
every digit, so the four communicating paths write identical bytes and the
number is fp8's own floor at these operands rather than anything the transport
adds. `gemm-only` is 0.0016583964824022402 against its local reference.

Validation reads the **received** buffer and checks every source rank's row
block separately against that rank's rebuilt operands. This matters more than
it sounds: during development the SDMA modes reported relL2 exactly 1.0 while
every *remote* slab was perfect and only the rank's own was missing -- the put
loop skips self, so a self-destination tile written to staging was stranded. A
check that sampled one remote slab, or that took a mean rather than a max, would
have passed it.

## Reproducing

```bash
# one cell
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_a2a/bench_gemm_a2a.py \
  --mode fused-sdma --chunks 4 -m 2048 -n 18432 -k 8192 \
  --warmup 8 --iters 21

# the whole table (three repeats per cell, retries the queue race)
python benchmark/cco/flydsl/gemm_a2a/sweep_a2a.py --out a2a_sweep.jsonl
```

`BUILD_CCO_SDMA=ON` is required for the SDMA modes. Without it every put is a
silent no-op; here that is caught -- the received slabs stay zero and validation
reports relL2 1.0 -- but only because validation reads what arrived.

Back-to-back launches lose the SDMA queue reclamation race (`anvil.cpp:237`)
often enough that a bare loop over the table cannot finish. The sweep driver
settles 25s between launches and retries a lost race after 90s rather than
recording it.

## Not measured

* `--quant blockscale`, which is what a model runs. The a2a shape here has no
  model behind it yet, so ptpc -- the kernel's native form -- is the honest
  default. `gemm_ar` found the fused margin *grew* under blockscale.
* Other `M`. gcnasm's rotation result is an `M >= 8192` effect and this table is
  `M = 2048`; the two do not settle each other.
* fp8 on the wire. Halving the payload is the obvious next lever and is the one
  thing here that would move the comm half again.
