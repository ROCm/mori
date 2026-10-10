# mori-SDMA A2A + bf16 GEMM (overlap example, AMD gfx950)

Standalone, self-contained example of overlapping an **All-to-All** collective with the
**bf16 GEMM** that consumes it, using **device-initiated SDMA** (mori `putmem_nbi_signal`)
so the transfer runs on the CU-free DMA copy engines while the GEMM runs on the CUs.

## What it does

Models a sequence-parallel `A2A -> projection` step (P = number of ranks):
- each rank holds `[world*sw, Hp]` (its tokens' local feature shard, `Hp = H/P`),
- an All-to-All redistributes it to chunk-major `a_full[P, sw, Hp]` (chunk `i` = source rank `i`),
- a split-K projection GEMM computes `out[sw, H] = sum_i a_full[i] @ W[i].T` (bf16).

Variants compared (2x2 comm-path x GEMM-structure, plus the in-kernel-gated fusion):
| variant | comm | GEMM |
|---------|------|------|
| `unfused_nccl` | nccl `all_to_all_single` | split-K (`world` per-source GEMMs) — production baseline |
| `unfused_mori` | mori-SDMA push+signal, wait all | split-K (no overlap) |
| `fused_ingrid` | mori-SDMA chunked push | split-K, per-source `gated_hgemm` with the SDMA wait fused INSIDE the GEMM (1 kernel, per-M-tile gate) |
| `unfused_mori_single` | mori-SDMA push + transpose | ONE big `[sw,H]@[H,H]` GEMM (seq-major) instead of split-K |
| `unfused_nccl_single` | nccl `all_to_all` + transpose | ONE big GEMM (isolates the single-GEMM lever on the nccl path) |

`unfused_mori` vs `unfused_mori_single` isolates split-K vs single-GEMM (same comm, no overlap);
`unfused_{nccl,mori}_single` isolates the comm path for the single-GEMM case. Correctness of every
variant is checked vs the `unfused_nccl` reference.

Measured (4-rank gfx950, H=8192, P=4, S=16384):
| | split-K | single-GEMM (transpose) |
|---|---|---|
| nccl | `unfused_nccl` 1.267 ms | `unfused_nccl_single` **1.122 ms (1.13x)** |
| mori | `unfused_mori` 1.304 ms | `unfused_mori_single` 1.172 ms (1.08x) |

`fused_ingrid` (mori, in-kernel-gated, split-K) ~= 1.29 ms (~ties nccl). Conclusions:
- The win is **single-GEMM (avoid split-K)**, not the comm path and not overlap: single-GEMM beats
  split-K on both comm paths, and `unfused_nccl_single` is the fastest overall.
- **mori SDMA adds nothing at this A2A size** — nccl beats it (small ~25 MB transfer, below the
  nccl/SDMA crossover). `fused_ingrid`'s in-kernel gate is diluted by split-K, so it only ties nccl.
- The single-GEMM path needs a transpose (mori `putmem` is contiguous-only), so it loses at small S
  where the transpose is not amortized.

## How the overlap works

1. **Producer** (`a2a_push_signal` / `a2a_chunked_push_signal`, `grid=P`): each block issues one
   peer's SDMA copy + an ordered signal (`flags += 1`) via `putmem_nbi_signal_block`, then returns
   (non-blocking). The transfers run on the **CU-free SDMA engines**, so they proceed while the CUs
   compute. Ordering is copy-then-signal, so a flag is set only *after* its data lands in HBM; the
   consumer reads the flag SYSTEM-scope (it is written by a remote SDMA signal).
2. **Overlap (`fused_ingrid`)**: the SDMA wait is compiled **inside** the GEMM (`gated_hgemm`,
   `ingrid=True`, `shard_rows=BLOCK_M`). Each GEMM M-tile spins on its own row-block flag before its
   A-load, so a tile computes as soon as its row-block lands — **while later row-blocks are still
   transferring** on the SDMA engines. One kernel, per-tile gate, no separate gate launches.
3. **No-overlap baselines** (`unfused_mori`): same CU-free SDMA push, but a separate `chunk_gate_signal`
   kernel waits for *all* chunks (+ barrier) before any GEMM runs — comm fully completes, then compute.

The FlyDSL SDMA producer + gate kernels are bundled in `a2a_gemm_example.py`, and the
forked in-kernel-gated GEMM in `flydsl_gated_hgemm.py` — fully self-contained (no external
project deps beyond the flydsl/mori/aiter runtime).

### In-kernel gate vs single-GEMM
- `fused_ingrid` uses `gated_hgemm` (`ingrid=True, shard_rows=BLOCK_M`): the SDMA wait is
  compiled INSIDE the GEMM — each M-tile spins on its own row-block flag before the A-load,
  so it is one kernel with per-tile gating. This is the design that wins for a *single large*
  gated GEMM (e.g. All-Gather + QKV). For this A2A O-proj it is split-K over `world` sources
  (many small gated GEMMs), so its per-tile-gate advantage is diluted — it ties nccl.
- `unfused_mori_single` avoids split-K entirely: transpose the chunk-major receive to seq-major and do
  ONE big `[sw,H]@[H,H]` GEMM. This wins (~1.12x @ S=16384) — but on GEMM structure, not
  overlap (the transpose is a barrier). mori `putmem` is contiguous-only, so seq-major needs
  that transpose; it therefore loses at small S where the transpose is not amortized.

## Run

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH=/opt/rocm/lib \
  MORI_ENABLE_SDMA=1 MORI_SHMEM_HEAP_SIZE=16G MORI_SOCKET_IFNAME=lo \
  torchrun --nproc_per_node=4 a2a_gemm_example.py --hidden 8192 --seqs 8192 16384
```

Requires: a gfx950 (MI350) node, the `flydsl` / `mori` / `aiter` runtime, and
`MORI_ENABLE_SDMA=1` (SDMA transport). Adjust `--hidden` and `--seqs` as desired.

## Notes / what to expect

- The SDMA path has a higher fixed overhead than nccl (push kernel + per-source gate
  kernels + a barrier) but higher bulk bandwidth. There is a crossover: nccl wins for
  small transfers, SDMA wins for large ones. For this small attention A2A (~25 MB) the
  overlapped `fused_ingrid` roughly ties nccl; the win comes instead from the single-GEMM
  structure (`unfused_mori_single`), and only at large S where its transpose is amortized.
- `bf16` is the "hideable" case (GEMM typically >= comm). Quantized (fp4) shrinks the
  GEMM below the comm, leaving less to hide.
