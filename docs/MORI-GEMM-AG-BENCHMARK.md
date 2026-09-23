# mori.ops.gemm_ag — fp8 GEMM fused with an all-gather

Measurements for `mori.ops.gemm_ag`, the row-concatenating all-gather fused into
the epilogue of mori's 8-wave fp8 GEMM.

**Headline: at every shape measured here, fusing an all-gather is slower than
not fusing it — and not because the fusion fails to overlap.** At
`wkv_gate-r4 M=512`, `fused-sdma --chunks 4` cuts the exposed communication from
50.9us to 21.8us, absorbing 57% of it. The transfer really is hidden. What
happens instead is that the epilogue's publication cost roughly doubles the
GEMM, from 53.6us to ~106us, which is more than the 29us it saved.

So there are two separate findings here and they should not be conflated:

* a **bound**, derived in *Why fusing does not pay*, which says all-gather has
  less to gain from fusion than either sibling operator and gets worse as the
  world grows; and
* a **cost**, the per-block whole-L2 writeback the fused epilogue needs before
  it may publish a chunk, which is the same line `gemm_a2a` calls "the single
  most expensive line in this epilogue". That one is an engineering target, not
  a law.

The operator is worth having today for its split paths, which beat RCCL. The
fused paths are documented as measured, not as hoped for.

## What the operator is

Every rank holds its own `A [M, K]` — its shard of the tokens — and a replicated
`B [N, K]`, computes the whole `C = A @ B.T` as `[M, N]` bf16, and every rank
ends up with all `world` of them concatenated along rows: `[world*M, N]`, source
rank `r` at rows `[r*M, (r+1)*M)`.

The shape it is aimed at is the DeepSeek V4-Pro prefill context-parallel
`wkv_gate` chain: `K = 7168` is the hidden size, `N` is 2048 on the ratio-4
layers and 1024 on the ratio-128 ones, and **`M = tokens/P`** — so sweeping `M`
is sweeping the sequence length. Each CP rank scores its own tokens and then
every rank needs every token's score.

| mode | what it does |
|---|---|
| `gemm-only` | the GEMM alone into a plain tensor, to size the ceiling |
| `gemm-to-window` | the same kernel, writing this rank's `recv` slot instead |
| `split-rccl` | `gemm-only()`; then `all_gather_into_tensor` |
| `split-lsa-push` | `gemm-to-window()`; a copy kernel vector-stores into every peer |
| `split-lsa-pull` | `gemm-to-window()`; a copy kernel loads from every peer |
| `fused-lsa` | C stored straight into every peer from the epilogue |
| `split-sdma` | `gemm-to-window()`; one copy-engine push per peer |
| `fused-sdma` | C into this rank's own slot, pushed per chunk from the epilogue |

Every mode above runs **the same GEMM**. `gemm-only` and the split paths run it
byte for byte — only the C pointer differs, which is what `gemm-to-window`
isolates — so the differences between them are transport and nothing else.

## How this differs from `gemm_a2a`, structurally

All-gather shards nothing, and three simplifications follow. They are worth
stating because they are also why the *fused* side has less to work with.

1. **No staging region.** `gemm_a2a` writes `[dst][M][shard_n]` in its epilogue
   purely so a copy engine has one contiguous source range per destination, and
   reserves a second full payload of window to hold it. All-gather's payload is
   the rank's whole `[M, N]`, already contiguous and already identical for every
   destination, so the GEMM writes straight into its own `recv` slot and that
   slab is pushed unchanged.
2. **One counter per chunk, not per (destination, chunk).** A chunk completing
   arms all `world-1` pushes at once.
3. **No tile rotation.** `gemm_a2a` and `gemm_ar` walk destinations round-robin
   so the last one's link does not idle until the end of the GEMM. A broadcast
   tile belongs to everyone, so the default `block_m`-outer order already gives
   every link chunk 0 at the same moment.

The price is on the wire: a rank sends `(world-1) * M * N * 2` bytes, **`world`
times what the all-to-all sends at the same `[M, N]`**. All-gather is the
bandwidth-bound member of the family.

## Conditions

8x MI355X (gfx950), fp8 e4m3 in / bf16 out, CUDA graph replay, median of 21
iterations after 30 warmups, max over ranks. Tile is 128x256 under `--quant
ptpc` and 256x256 under `--quant mxfp8`, which needs `BLOCK_M=256` for the scale
operands. Every launch waits for an idle box; any sample taken with a neighbour
present is discarded rather than averaged in, and clock and junction temperature
are recorded next to every measurement. That protocol is `gemm_a2a`'s and its
rationale is in that file — a neighbour is worth 9% on a kernel that
communicates nothing.

## Results

<!-- FILLED FROM sweep_models.py -->

## Why fusing does not pay

Producer-side fusion overlaps the transfer with the compute that produces it, so
the best it can do is replace `G₀ + C₀` with `max(G₀, C₀)`. Its ceiling is
therefore

    saving ≤ min(G₀, C₀) / (G₀ + C₀)

and for an all-gather that ratio is fixed by the shape, not by the code:

    G₀ ∝ 2·M·N·K / FLOPS        C₀ ∝ M·N·(world−1)·2 / BW

    G₀ / C₀  ∝  K · BW / ((world−1) · FLOPS)

`M` and `N` cancel. **The only levers are `K` and `world`** — more reduction
depth per output element, or fewer peers to broadcast to. At `K = 7168` and
`world = 8` the measured ratio is `57 / 155`, so the ceiling on any fusion of
this collective at this shape is 27%, and the fused epilogue's own cost exceeds
it.

Contrast the sibling operators, which is the useful part:

| | bytes on the wire per rank | `G₀/C₀` scales as | measured fused gain |
|---|---|---|---|
| `gemm_ar` (all-reduce) | `2·M·N/world` per leg | `K·world` | absorbs ~37% of comm |
| `gemm_a2a` (all-to-all) | `M·N·(world−1)/world` | `K·world` | 3.5–15.9% end to end |
| `gemm_ag` (all-gather) | `M·N·(world−1)` | `K/world` | negative |

All-gather is the only one of the three where **adding ranks makes fusion
*less* attractive**: every extra rank adds a full copy of the payload to the
wire while adding nothing to the compute. The other two shard the payload by
`world`, so scaling out leaves the ratio flat or improves it.

This is the same structural reasoning as the "post-barrier phase" account of why
all-reduce absorbs 37% and an EP combine absorbs 81%, applied one level up: that
one asks what fraction of the collective is fusable, this one asks whether there
is enough compute to fuse it into.

## Reading the tables

<!-- FILLED FROM sweep_models.py -->

## A correctness note worth keeping

`split-lsa-pull` shipped in a first draft with the producer's release missing,
and **it validated**: relL2 1.66e-3 under `--quant ptpc`, the same figure every
other mode reports. Under `--quant blockscale` it gave 2.05e-1. The blockscale
GEMM is 96us against ptpc's 57, so the rank skew is wider and the window the
race needs is wider with it.

The cause is specific to the pull direction and is a real constraint on it: a
pull reads a peer's HBM over xGMI, while the peer's GEMM left C dirty in its own
L2. Nothing downstream can repair that. `cco_system_fence` is
`__threadfence_system()`, which orders the *calling thread's own* prior writes —
so a fence in the copy kernel, which wrote nothing, publishes nothing. The
producer has to publish, and the producer is the GEMM: the pull path compiles it
with `peer_uncached=True`, whose epilogue stores through a buffer descriptor
with `sc0|sc1`. At this shape that costs nothing measurable (217.0us against
217.1us before the change).

The general lesson is the one the mode matrix is built around: **a transport bug
that only one quantisation exposes is not found by spot-checking the default.**
`tests/python/cco/test_gemm_ag.py` parametrises every mode over all three.

## Shape constraints

`validate()` requires `n % block_n == 0` and `m % block_m == 0`, and that is all.
`gemm_a2a` additionally requires `n % (world_size * block_n) == 0` — **N a
multiple of 2048 at P=8** — because a destination's column shard has to be a
whole number of GEMM tiles. No column is sharded here, so that rule does not
arise.

That is not a corner case. The `wkv_gate` ratio-128 layers are `N = 1024`, which
`a2a_config` rejects at `world_size=8` and `ag_config` accepts.

| shape | K | N | `ag_config` | `a2a_config` |
|---|---:|---:|---|---|
| wkv_gate ratio-4 | 7168 | 2048 | ok | ok |
| wkv_gate ratio-128 | 7168 | 1024 | ok | **rejected** |
| Llama-3.1-70B | 8192 | 10240 | ok | ok |
| Llama-3.1-405B | 16384 | 18432 | ok | ok |

`--quant mxfp8` adds `N % 32 == 0`, `K % 128 == 0` and `BLOCK_M == 256`, all
from the ue8m0 operand format rather than from this operator.

## Reproducing

```bash
# one cell
MORI_ENABLE_SDMA=1 MORI_SOCKET_IFNAME=lo \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ag/bench_gemm_ag.py \
  --mode split-lsa-pull -m 2048 --out-dim 2048 -k 7168 \
  --warmup 30 --iters 21 --phase-split

# the model tables, with the idle-box protocol
python benchmark/cco/flydsl/gemm_ag/sweep_models.py --rounds 3 --out models.jsonl
python benchmark/cco/flydsl/gemm_ag/sweep_models.py --quant mxfp8 --rounds 2 \
  --models wkv_gate-r4 --out mxfp8.jsonl

# the knob sweep at one shape: push vs pull, unroll, cache policy, chunks
python benchmark/cco/flydsl/gemm_ag/sweep_ag.py --out sweep.jsonl

# layout arithmetic, no GPU
pytest tests/python/cco/test_gemm_ag.py -k "not validates and not deterministic"
```

`-n` is `--out-dim` here, not `-n`: `torchrun` takes `--nproc_per_node` and
argparse's abbreviation matching makes a bare `-n` ambiguous against it.

`BUILD_CCO_SDMA=ON` **and** `MORI_ENABLE_SDMA=1` are required for the SDMA
modes. Without the environment variable every put is a silent no-op — the run
completes, the timing looks plausible because it is the GEMM plus a barrier, and
every peer slot keeps whatever it held. The benchmark now refuses to start the
SDMA modes without it rather than relying on validation to notice.

Back-to-back launches lose the SDMA queue reclamation race (`anvil.cpp:237`)
often enough that a bare loop cannot finish a table. Both drivers settle between
launches and retry a lost race rather than recording it.

## Not measured

* Cross-node (GDA). Everything here is intra-node LSA/SDMA over xGMI.
* fp8 on the wire. This is the direction where it would help most — the
  operator is bandwidth-bound by construction and a 2x narrower wire moves the
  `G₀/C₀` ratio directly — and the `pull` direction is built so the dequantize
  lands in registers on the consumer, which is where `gemm_ar` measured it as
  free against a 61.0us widen kernel on the SDMA push. Doing it is the obvious
  next step and is worth more here than any amount of epilogue tuning.
* `world != 8`. The `G₀/C₀` derivation says fusion gets *more* attractive as
  `world` falls, so 2 and 4 ranks are the regime where the fused paths might
  come back.
* `--chunks` past 8.
