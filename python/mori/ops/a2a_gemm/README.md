# `mori.ops.a2a_gemm` -- all-to-all fused into the fp8 GEMM that consumes it

Ulysses sequence parallelism ends attention with every rank holding all
`world * m_local` tokens for its own `K / world` columns of heads. The
o-projection needs the opposite -- this rank's `m_local` tokens over all `K` --
times the replicated weight `B [N, K]`:

    C_d = concat_s(A_s[d*m_local:(d+1)*m_local]) @ B.T     # [m_local, N]

where source `s` owns columns `[s*k_shard, (s+1)*k_shard)` of the concatenation.
Split, that is `all_to_all_single` then a GEMM. Fused here, the GEMM starts on
this rank's own columns while the copy engines move everyone else's, and each
remote K segment is read as soon as its band of rows has landed. It is the
reverse of `mori.ops.gemm_a2a` (#695), which fuses the QKV projection's GEMM
*into* the exchange that follows it.

## How it works

The window keeps what this rank sends and what it receives side by side, in the
same per-source layout (`layout.py`):

    [ control | send: [world][m_local][k_shard] fp8 | recv: [world][m_local][k_shard] fp8 ]

`send[d]` -- the row block bound for rank `d` -- is one contiguous slab, so the
head shard goes out as one put per destination and band. The GEMM reads A one K
segment at a time from wherever it lives: `send[rank]` for this rank's own
columns, which never travel, and `recv[s]` for every other source. Only the A
addressing changes in the GEMM (row stride `k_shard`, a base per segment); the
8-wave main loop is `gemm_ar`'s, untouched.

1. **Block 0 is the poster.** Lanes `0..world-1` each own a peer and push it
   `send[peer]` in `chunks` row bands, back to back. Each put carries a trailing
   copy-engine atomic that bumps the peer's signal counter for this rank: same
   queue, after the bytes, so seeing the count is seeing the band. Block 0 then
   computes its tile and only *then* `quiet`s, which keeps the rank from reusing
   `send` while bytes are on the wire without putting the transfer in front of
   that tile.
2. **K order.** A tile walks K starting with this rank's own segment, then the
   other sources rank-rotated, so it has `1/world` of its work before anything
   needs to arrive. Tiles go band-major: the first wave needs only band 0.
3. **The wait** (`wait=`). `segment`: every wave spins on a segment's band just
   before its first A load of it. `tile`: thread 0 waits for the band from every
   source before the prologue. Block 0 takes the last tile, which needs the last
   band anyway, so posting costs it nothing.
4. **A barrier-only kernel closes every call** and advances the epoch the
   counts are compared against: nobody may refill a peer's `recv` while that
   peer still reads it.

The signal counters are never reset, so `build_phases()["arm"]` snapshots them
into the window once, before the first call; a band has landed when its source's
counter reaches `snapshot + (epoch - 1) * chunks + band + 1`.

**ptpc, with a scale over all heads.** The A scale is one fp32 per token over
the whole K, so it is computed before quantization from the row maxima gathered
across ranks -- what ATOM's fp8 head exchange (ROCm/ATOM#2364) already does --
and is local to the receiver. It never travels.

### Three things that cost more than the overlap saved

The first two were measured on the M3 shape below, `m_local=2048`, one round;
the third on the 70B shape.

* **A full system fence after each wait.** `cco_system_fence` is
  `__threadfence_system`, whose release half writes back L2 -- dirty with the
  kernel's own C stores -- and every wave issued one per remote segment. Fused
  `c1` took 187.5us against 132.4us for `split-sdma`. An acquire-only fence
  (`buffer_inv`) brought it to 151.8us.
* **`quiet` and a flag store per band.** The first poster drained each band
  before posting the next, so the copy engine idled between bands and more
  chunks meant *slower*. Per-band remote signals let the bands stream:
  `m_local=4096 c2` went from 199.1 to 190.2us, and the chunk count stopped
  hurting at large M.
* **Block 0 draining before its own tile.** The poster `quiet`ed right after
  posting, so block 0 started its tile only once the whole transfer had landed:
  the transfer *plus* a full tile on the critical path. With the waits switched
  off (`--wait none`, a diagnostic) the fused kernel was a flat 51us over
  `gemm-only` at both `m_local=4096` and 8192 at P=8; deferring the quiet past
  the tile took that to 9us at 4096, and took M3's best cells 2-4% lower.

`acquire="none"` drops the invalidate as well and is what the tables use. It is
correct because cco allocates windows uncached by default
(`CCO_UNCACHED_WINDOW`): A's loads from the window never hit a cached line, so
there is no stale one to invalidate, and they cannot pass the spin's loads. With
`CCO_UNCACHED_WINDOW=0` use the default, `acquire="inv"`.

## Measured

fp8 ptpc in, bf16 out, 256x256 tile. MI355X, one node (`crsuse2-m2m-v3-095`,
2026-10-28), FlyDSL 0.4.0, torch 2.12.0+rocm7.14.0. Every case runs in one
process on one communicator; a case replays its CUDA graph back to back with an
event between launches for at least 3 s and reports the median launch gap, max
over ranks; three rounds interleaved (spread 0.0-2.0%). Every case validated
every row of C at the fp8 floor (relL2 1.66e-3) against an fp32 reference, for
two different operand sets. `--acquire none` throughout.

`split-rccl` is `all_to_all_single` and the same GEMM; `split-sdma` is push,
barrier, the same GEMM, barrier. Times in us.

**MiniMax-M3 o-projection at Ulysses SP4** -- `K = N = 6144`, P=4,
`k_shard = 1536`:

| m_local | gemm-only | split-rccl | split-sdma | best fused | vs split-sdma | vs RCCL |
|---:|---:|---:|---:|---|---:|---:|
| 2048 | 63.0 | 144.0 | 132.9 | **121.7** `segment c1` | -8.5% | -15.5% |
| 4096 | 125.1 | 264.9 | 244.0 | **184.1** `segment c2` | -24.5% | -30.5% |
| 8192 | 208.6 | 479.0 | 438.7 | **300.2** `segment c4` | -31.6% | -37.3% |

**Llama-3.1-70B o-projection at P=8** -- `K = N = 8192`, `k_shard = 1024`:

| m_local | gemm-only | split-rccl | split-sdma | best fused | vs split-sdma | vs RCCL |
|---:|---:|---:|---:|---|---:|---:|
| 2048 | 94.3 | 156.2 | 145.8 | **134.4** `segment c1` | -7.8% | -14.0% |
| 4096 | 182.6 | 294.6 | 269.6 | **258.6** `tile c2` | -4.1% | -12.2% |
| 8192 | 356.4 | 552.5 | 506.4 | **435.2** `tile c8` | -14.0% | -21.2% |

Every fused configuration:

| shape | m_local | wait | c1 | c2 | c4 | c8 |
|---|---:|---|---:|---:|---:|---:|
| M3 P=4 | 2048 | segment | **121.7** | 128.2 | 136.8 | 157.5 |
| | 2048 | tile | 128.8 | 136.6 | 144.6 | 165.5 |
| | 4096 | segment | 234.0 | **184.1** | 190.8 | 209.4 |
| | 4096 | tile | 240.7 | 188.1 | 198.8 | 217.0 |
| | 8192 | segment | 424.0 | 356.1 | **300.2** | 317.6 |
| | 8192 | tile | 434.7 | 363.7 | 307.6 | 326.2 |
| 70B P=8 | 2048 | segment | **134.4** | 139.0 | 149.0 | 176.0 |
| | 2048 | tile | 141.8 | 141.8 | 151.6 | 179.0 |
| | 4096 | segment | 262.4 | 259.7 | 261.4 | 263.0 |
| | 4096 | tile | 264.0 | **258.6** | 259.0 | 259.7 |
| | 8192 | segment | 515.2 | 499.8 | 466.0 | 448.3 |
| | 8192 | tile | 506.9 | 482.9 | 452.8 | **435.2** |

**Reading it.**

- **Fusing wins in all six cells**, 4-32% against `split-sdma` and 12-37%
  against RCCL. It wins most where the transfer is the larger half: at P=4 the
  exchange (`split-sdma - gemm`) is 70-230us against a 63-209us GEMM.
- **The chunks are the mechanism, again** (as in `ag_gemm`). With one chunk a
  remote segment waits for a peer's whole slab and the gain is a few percent;
  bands let a wave start after a fraction of the transfer. At P=4 the best count
  rises with M -- 1, 2, 4 -- and smaller bands lose more to per-put bandwidth
  than they gain.
- **`segment` is the default to reach for.** It is best or within 0.5% in five
  of six cells, and wins whenever the tiles fit in one wave (`m_local=2048`):
  there, starting on the own segment is the only overlap there is. `tile` is
  ahead only at P=8, M >= 4096, by 0.4-3%.
- **At P=8 the transfer slows down under the GEMM.** With the waits switched
  off (`--wait none`) the fused kernel ends within 9us of `gemm-only` at
  `m_local=4096`, so all seven slabs land inside the GEMM; yet with the waits
  on, the second wave of tiles sits on the last bands for ~70us, which fits a
  transfer running at roughly half its idle rate (an inference from the two
  times, not a measurement of the copy engines). This is why 70B gains
  4-14% where M3 gains 8-32%. Not a polling cost -- an `s_sleep` backoff in the
  spin changed nothing -- and not the window's cache mode: a cached window
  (`CCO_UNCACHED_WINDOW=0`, `acquire="inv"`) measured the same in `tile` mode.
  What the copy engines contend with is not yet identified.

## Running it

```bash
# M3 at P=4 (the 70B table: --nproc_per_node=8, -n 8192 -k 8192)
HIP_VISIBLE_DEVICES=0,1,2,3 MORI_SOCKET_IFNAME=lo MORI_ENABLE_SDMA=1 \
  torchrun --standalone --nproc_per_node=4 \
  benchmark/cco/flydsl/a2a_gemm/bench_a2a_gemm.py -m 2048 4096 8192 \
  -n 6144 -k 6144 --chunks 1 2 4 8 --wait segment tile --acquire none \
  --rounds 3 --out m3.jsonl

MORI_ENABLE_SDMA=1 pytest -q tests/python/cco/test_a2a_gemm.py
```

`--wait none` and `--no-post` are diagnostics (no waits, so not validated; no
puts): they split a fused time into what the poster costs and what the waits
cost. `--back-to-back R` adds R calls with no host sync between them, each with its
own operands and output, one rank held on the GPU before each, and validates
every one.

## Not done yet

Why the P=8 transfer halves under the GEMM (a timeline, as `gemm_a2a --timeline`
has); a public op wrapper with `self_test` and a per-M tile/chunk choice; blockscale
and mxfp8; N that is not a multiple of 256; fp8 head shards quantized against a
per-shard scale (the scale has to be global, see above); a model integration.
