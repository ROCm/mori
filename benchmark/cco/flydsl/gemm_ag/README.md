# GEMM + All-Gather benchmarks

These entry points call the kernels in `mori.ops.gemm_ag` directly. They do
not generate source code or require a pinned Git revision. Use a ROCm/FlyDSL
environment with MORI's CCO host library built with SDMA support.

M is the number of rows **per rank**. Each rank computes `[M,K] @ [N,K].T`,
then receives the concatenated `[world_size*M,N]` result. The BF16 comparisons
below use FP32 accumulation, output and communication.

## Environment

Run from the repository root, using its installed Python environment:

```bash
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export MORI_ENABLE_SDMA=1
export BUILD_CCO_SDMA=1
export MORI_SOCKET_IFNAME=lo
```

Run configurations sequentially on idle GPUs. The commands below use eight
ranks; adjust `--nproc_per_node` for another supported world size. Allow SDMA
queues to tear down before launching the next process group.

## Single-stream GEMM + AG

The non-fused baseline computes the complete local result before its SDMA
gather. The fused kernel submits chunks from the GEMM epilogue and finishes
with a drain. Both launch their kernels on one stream.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ag/bench_gemm_ag.py \
  --mode split-sdma --in-dtype bf16 --out-dtype fp32 \
  -m 8192 --out-dim 2048 -k 7168 --block-m 128 --block-n 128 \
  --rounds 5 --warmup 50 --iters 101 --tolerance 1e-5

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ag/bench_gemm_ag.py \
  --mode fused-sdma --chunks 4 --peer-uncached --fence release \
  --in-dtype bf16 --out-dtype fp32 \
  -m 8192 --out-dim 2048 -k 7168 --block-m 128 --block-n 128 \
  --rounds 5 --warmup 50 --iters 101 --tolerance 1e-5
```

For the M sweep, run each command with M=4096/8192/16384 and try fused chunk
counts 1/2/4/8. `--peer-uncached --fence release` selects the publication
optimization previously called `sc_release`; omit both for the original
cached-store/leader-fence configuration. Defaults are unchanged.

`--torch-gemm` uses native `torch.mm(..., out_dtype=torch.float32)` for
BF16/FP32 `gemm-only`, `split-rccl` or `split-sdma` controls. The other existing
transport modes, FP8 precision/quantization options and phase diagnostics
remain available through `--help`. `--wait-policy` is BF16-only. A `--no-put`
diagnostic does not perform a complete all-gather.

## M-chunk pipeline

This path uses a compute stream and a submission stream. Each chunk's ready
event follows its GEMM and, for split-K, FP32 reduction. SDMA reads the final
chunk output in the registered window. The final drain waits for all copies
and the cross-rank barrier.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ag/bench_gemm_ag_pipeline.py \
  -m 8192 --out-dim 2048 -k 7168 \
  --backend mori --chunks 4 --schedule overlap \
  --rounds 5 --warmup 50 --iters 101
```

Choose `--backend splitk --split-k 4` for the four-way K partition and
reduction, or `--backend torch` for native Torch chunk GEMMs. `--schedule
serial` retains the two streams but waits for all chunk computations before
submitting any transfers; it is a control without compute/transfer overlap.
Each M chunk and N must be divisible by 128. Each K partition must contain
at least two complete 64-element steps.

## Pure GEMM comparison

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmark/cco/flydsl/gemm_ag/compare_gemm.py \
  -m 2048 --out-dim 2048 -k 7168 --rounds 5 --warmup 100 --iters 101
```

This compares native BF16-to-FP32 Torch with the retained MORI tile choices,
rotating case order across rounds. It emits `COMPARE_JSON` with per-rank data.

## Measurement and validation

`bench_gemm_ag.py` and the pipeline emit `RESULT_JSON` and `TIMING_JSON`.
With `--rounds 5 --warmup 50 --iters 101`, the reported time is the median of
five per-round maxima across ranks, each taken from a rank's 101-sample
median. Allocation, compilation, capture and reference computation are
outside timing. `_bench_utils.py` implements the shared timing protocol.

BF16 runs validate initial output and two changed-input replays, cloning the
received tensor immediately before constructing references. The pipeline
also poisons partial storage between replays. Nonfinite peer output fails
validation. FP8 transport runs retain their initial precision-specific check.

The [benchmark report](../../../../python/mori/ops/gemm_ag/README.md) and
[optimization report](../../../../python/mori/ops/gemm_ag/EXPERIMENTS.md) retain
the measured findings. Historical generators, run plans and raw datasets are
archived separately from the maintained source tree.

## Kernel interfaces

| Interface | Purpose |
|---|---|
| `compile_bf16_gemm_ag(..., split_k=1)` | Existing BF16 GEMM and fused AG paths |
| `compile_bf16_gemm_ag(..., split_k=2/4/8, out_dtype="fp32")` | Unfused local partials in `[split_k,M,N]`; caller performs reduction |
| `build_sdma_chunk_post(cfg, rank)` | Submit a completed contiguous chunk to each peer on queue 0 |
| `build_sdma_phases(cfg, rank, queues=1)["drain"]` | Finish the serialized chunk submissions and cross-rank synchronization |

Split-K preserves the A/B row stride K. It requires local FP32 partials;
`fuse=True` and peer publication are rejected for that output. Chunk posts
must be serialized on their submission stream. Fused GEMM producers instead
retain their existing per-chunk queue assignment and all-queue drain.
