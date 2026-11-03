# Example 13 -- MegaMoE stage1 dispatch on cco SDMA (gfx950, prototype)

aiter's `MegaMoEV2` stage1 copies every routed row to its destination rank with P2P stores from
producer CUs. Here the sender quantises the rows into a local send buffer `S` grouped by
destination rank, the copy engines ship each run with SDMA `put`, and GEMM1 reads its A rows
(and their e8m0 scales) through an index table `rowmap`. It is the stage1 counterpart of
example 11 (stage2 + combine). The stage1 outputs are bit for bit aiter's.

| file | what |
|---|---|
| `stage1_sdma.py` | `Arena` (window layout), `build_prepare` (one resident kernel: plan, route push, quantise into `S`, put the first chunks), `post_rest_chunks`, `build_post_rest`, `build_wait`, host `Stage1SdmaPrepare` |
| `gemm1_sdma.py` | GEMM1 on aiter's `do_tile`, A and scales from `[S \| R]` through `rowmap`; with `prep` its workgroup 0 puts the remaining chunks and the others wait for the chunks they need |
| `test_prepare.py` | the rows each rank receives equal aiter's quantisation, byte for byte (no aiter dispatch needed) |
| `test_dispatch.py` | `R[rowmap[r]]`, weights and srcmap equal aiter's `rx_em`, `wts_em`, `srcmap_em` row by row |
| `test_gemm1.py` | stage1 output and the final layer output equal aiter's; `--stress N` repeats a poisoned call; `--time-iters` times it |
| `baseline.py` | where aiter's own prepare / stage1 / stage2 time goes |

Design in short:

* a token is sent to a rank once, however many experts it has there; its row sits at the
  position of its FIRST expert on that rank, so a landed prefix of a run is a prefix of experts;
* `prepare` puts the first `--early-chunks` chunks of each run, the GEMM1 comm workgroup the
  rest (posting puts in `prepare` cost ~50 us of its critical path);
* a GEMM1 tile waits only for the chunks its expert needs (`need`, SDMA signal counters);
* no cache invalidation per tile: a chunk starts on a 128 B line (`chunk_rows % 8 == 0`).

```bash
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/13_flydsl_mega_moe_stage1_sdma/test_gemm1.py --network v4_pro --bs 4096 \
  --chunk-rows 256 --early-chunks 2 --stress 50 --time-iters 30
```

## Status -- read before relying on it

* Needs aiter with `MegaMoEV2`'s compact prepare (its plan tables are reused for the route
  push), FlyDSL, a gfx950 node with 8 GPUs, mori built with SDMA.
* The speedups measured (1.03-1.15x on prepare + stage1, eager, v4_pro and glm52) are against
  aiter main WITHOUT ROCm/aiter#7001, and on a node whose GPUs were shared with other jobs.
  Against #7001 this is probably behind at bs 4096 and 8192 (see ROCm/mori#904, which reports the
  same for its SDMA stage1). It has not been measured.
* One call at a time. Nothing yet protects `S` and `R` between back-to-back calls (the tests
  put a barrier between calls), and the SDMA signal / flag totals are kept by the host.
* Not done: GEMM1 does not do this rank's tokens first (a tile mixes all source ranks; the
  measured wait for data is under ~25 us, so there is little to hide), and it is aiter's GEMM1
  tile, not the 8-wave 256x256 one.
