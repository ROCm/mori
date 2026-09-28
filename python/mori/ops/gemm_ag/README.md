# GEMM + All-Gather

`mori.ops.gemm_ag` computes a local GEMM and gathers the results along the row
dimension. Each rank holds `A [M,K]` and a replicated `B [N,K]`, computes
`C = A @ B.T`, and receives the concatenated `[world_size*M,N]` result.
The implementations support FP8 and BF16 inputs, with LSA and SDMA transport
paths.

- [Benchmark and correctness report](MORI-GEMM-AG-BENCHMARK.md): operator design,
  precision and transport comparisons, BF16 synchronization fixes, and validation.
- [Fusion optimization report](MORI-GEMM-AG-OPT-EXPERIMENTS.md): publication and
  chunking experiments, split-K and two-stream pipelines, and larger-M results.
- [Running the benchmarks](../../../../benchmark/cco/flydsl/gemm_ag/README.md):
  environment setup, commands, timing protocol, and kernel interfaces.
