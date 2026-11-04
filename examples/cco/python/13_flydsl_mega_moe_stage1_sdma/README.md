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
| `test_gemm1.py` | stage1 output and the final layer output equal aiter's; `--stress N` repeats a poisoned call; `--time-iters` times it; `--trace N` prints the clock stamps below; `--stall-test` runs GEMM1 twice after one prepare |
| `baseline.py` | where aiter's own prepare / stage1 / stage2 time goes |

Design in short:

* a token is sent to a rank once, however many experts it has there; its row sits at the
  position of its FIRST expert on that rank, so a landed prefix of a run is a prefix of experts;
* a token's bf16 row is read and quantised ONCE and stored to its row in every run it is in
  (a token goes to ~4.6 ranks, so the input is read once instead of ~4.6 times);
* `prepare` puts the first `--early-chunks` chunks of each run, the GEMM1 comm workgroup the
  rest (posting every chunk in `prepare` cost ~50 us of its critical path);
* a grid rendezvous separates the quant pass that fills the early chunks from the rest, so the
  first puts are not held back by the second pass racing them for bandwidth (it moved the first
  put from 177 to 106 us and the first tile's wait for data from 53 to 7 us at bs 8192);
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
* The speedups measured are against aiter main WITHOUT ROCm/aiter#7001, eager, and on a node
  whose GPUs were shared with other jobs (so the cleanest of several runs is quoted).
  Against #7001 this is probably behind at bs 4096 and 8192 (see ROCm/mori#904, which reports the
  same for its SDMA stage1). It has not been measured. Flash (model_dim 5120, inter 2304, 384
  experts, topk 6), prepare + stage1, 8x MI355:

  | tokens/rank | 512 | 1024 | 2048 | 4096 | 8192 |
  |---|---|---|---|---|---|
  | `--chunk-rows` / `--early-chunks` | 128 / 4 | 128 / 2 | 128 / 2 | 256 / 2 | 256 / 2 |
  | vs aiter main | 1.36x | 1.17x | 1.14x | 1.06x | 1.00x |

  The gain is the dispatch overhead, which is roughly fixed, over a GEMM1 that grows with the
  token count: that overhead is 41% of the call at bs 512 and 18% at bs 8192 (`--trace`).
* One call at a time. Nothing yet protects `S` and `R` between back-to-back calls (the tests
  put a barrier between calls), and the SDMA signal / flag totals are kept by the host.
* Not done, and measured not to be worth it: GEMM1 does not do this rank's tokens first. A tile
  mixes all source ranks, so a local-first tile has to be padded -- this rank owns 1/8 of an
  expert's rows against a 64 or 128-row tile -- and the padding costs more GEMM1 time than the
  wait it removes at every token count. Smaller chunks do not help either: they move the first
  put earlier but the comm workgroup then needs 197 us instead of 56 to post them all (bs 8192).
* Not done: GEMM1 is aiter's tile, not the 8-wave 256x256 one, which alone measured 1.18-1.33x
  faster but wants contiguous A rows and pre-shuffled A scales.
* `--trace` stamps the 100 MHz clock inside both kernels (off by default): per prepare block
  [start, count done, plan done, end] and when each early put was posted; per GEMM1 workgroup
  its start, start gate, end, and per work item (gate start, gate passed, item done).
