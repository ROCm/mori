# CCO examples

Runnable examples for the **cco** GPU-communication API — both Python (FlyDSL +
the cco host runtime) and C++.

```
examples/cco/
├── python/   # FlyDSL device kernels + cco host runtime (mpi4py bootstrap)
├── ir/       # Triton device kernels + cco host runtime (torchrun bootstrap)
└── cpp/      # standalone C++ host+device examples (MPI bootstrap)
```

| Example | Lang | Shows |
|---|---|---|
| `python/01_barrier` | py | cco host barrier across ranks |
| `python/02_lsa_put` | py | intra-node LSA put via a hand-written `.hip` kernel (mori.jit) |
| `python/03_flydsl_put` | py | FlyDSL GDA put + signal/wait |
| `python/04_flydsl_lsa_put` | py | FlyDSL LSA: direct peer-pointer store in the kernel |
| `python/05_flydsl_lsa_allreduce` | py | FlyDSL LSA custom all-reduce (peer pointers + device signal barrier) |
| `python/06_flydsl_gda_modes` | py | FlyDSL GDA template matrix: (thread_mode, coop) × signal, plus DEFERRED / CALLER_ORDERED handles with flush_async + wait |
| `python/07_flydsl_sdma` | py | FlyDSL SDMA put/get over the copy engine |
| `python/08_torch_symm_import` | py | import a `torch.symm_mem` tensor into the flat LSA space, no copy |
| `python/09_torch_symm_lsa_all2all` | py | mori's own torch symm-mem backend + Triton `lsa_ptr` all-to-all |
| `python/10_flydsl_sdma_allgather_async` | py | FlyDSL SDMA all-gather split into `start_async` / `wait_async`, overlapped with a GEMM |
| `python/11_flydsl_mega_moe_sdma` | py | MegaMoE stage2 + combine (gfx950) with the transfer done by SDMA: GEMM2 writes a rank-sorted send buffer and comm waves (one per peer) inside the GEMM kernel `put` it; `mega_moe_gemm2_with_sdma.py` is the op and its kernels, `gemm2_8wave.py` an 8-wave persistent GEMM2 for it (`--gemm2 8wave`), `main.py` its test and benchmark against aiter (needs aiter) |
| `python/11_flydsl_mega_moe_sdma/sdma_bandwidth.py` | py | what one cco SDMA channel moves, with no GEMM in the way |
| `python/12_flydsl_sdma_allgather_engine_wait` | py | In-place all-gather over run-time regions; the recommended `post_wait_mem_put_ptr` use: read each peer's READY once, plain put if it is there, else the copy engine (not a CU) waits for it |
| `python/13_flydsl_mega_moe_stage1_sdma` | py | MegaMoE Stage1 (dispatch + GEMM1, gfx950) with the dispatch done by SDMA: a pack kernel lays out this rank's rows per (destination, expert), then one fused kernel puts them (workgroup 0, one signalled put pair per peer and expert) while an 8-wave GEMM1 starts on each expert as soon as its rows land; `mega_moe_stage1_sdma.py` is the op (installs into aiter's `MegaMoEV2` in place of its Stage1 launch), `gemm1_8wave_sdma.py` the fused kernel, `main.py` its test and benchmark against aiter (needs aiter) |
| `ir/test_triton_cco.py` | py | Triton DevComm queries and LSA/SDMA/GDA device APIs |
| `cpp/01_lsa_put.cpp` | c++ | intra-node LSA put (includes only `cco.hpp`) |
| `cpp/02_gda_put.cpp` | c++ | GPU-initiated RDMA put + signal/wait (includes only `cco_scale_out.hpp`) |

GDA examples move data over RDMA (cross-node capable). LSA examples are
intra-node only: cco hands the kernel the peer's load/store-accessible VA and the
kernel writes it directly.

---

## 1. Set up the environment

Use the **`deploy-mori`** skill (`.claude/skills/deploy-mori`) — it starts the
container with the right device/NIC mappings, installs ROCm + NIC userspace
libraries (AINIC / ConnectX / Thor2/BNXT) + RDMA-core, and installs MORI. The
core of it is:

```bash
# inside the MORI container, at the repo root
pip install pybind11 -q          # build dependency missing from pyproject
rm -rf build                     # clear any stale cmake cache
pip install .                    # builds + co-locates all libmori_*.so
```

Then install the two extra runtime deps the examples need (not pulled in by
`pip install .`):

```bash
pip install mpi4py "flydsl==0.4.0"
```

- `mpi4py` — every example bootstraps the cco `UniqueId` over MPI.
- `flydsl` — required by the Python **FlyDSL** examples (03–08). 0.4.0 is what CI
  and aiter use; 0.2.x and 0.3.x work too. (Also available as the optional extra
  `pip install amd_mori[flydsl]`.) Not needed for `01`, `02`, `09`, or the C++
  examples.

After `pip install .` you do **not** need `PYTHONPATH` / `LD_LIBRARY_PATH` /
`MORI_CCO_BC`: the shared libs are co-located in `site-packages/mori/` (RUNPATH
`$ORIGIN`) and the FlyDSL device bitcode is JIT-compiled on first use.

The only env var needed at run time is the RDMA interface:

```bash
export MORI_SOCKET_IFNAME=<iface>     # e.g. enp159s0np0; see `ls /sys/class/net`
```

---

## 2. Run the Python examples

```bash
cd <repo>
export MORI_SOCKET_IFNAME=<iface>
export MORI_CCO_GDA_CONN=full          # required for GDA on a single node (03/06)

mpirun --allow-run-as-root -n 2 python3 examples/cco/python/01_barrier/main.py
mpirun --allow-run-as-root -n 2 python3 examples/cco/python/03_flydsl_put/main.py
# ... 02, 04, 05, 06 likewise
```

Example 10 bootstraps with `torchrun` instead of MPI and needs SDMA:

```bash
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/10_flydsl_sdma_allgather_async/main.py
```

Example 11 also uses `torchrun` and SDMA, and additionally needs aiter (it reuses
`MegaMoEV2` for the reference and the GEMM2 mainloop) on a gfx950 node with 8 GPUs.
`--gemm2` picks the GEMM2 kernel: `aiter` (aiter's GEMM2 main loop with comm
workgroups in front) or `8wave` (an 8-wave 256x256 persistent GEMM2 whose
workgroup 0 is the comm workgroup; fp8, `remote` signalling, one queue -- the
defaults). Both share the plan, the window and the combine:

```bash
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/11_flydsl_mega_moe_sdma/main.py test --network glm52 --bs 4096 --gemm2 8wave

# latency vs aiter's MegaMoEV2 (mori.shmem P2P scatter), sweeping token counts
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/11_flydsl_mega_moe_sdma/main.py bench --network glm52 \
  --bs-list 128,512,2048,4096,8192,16384 --no-aligned-pair --gemm2 aiter,8wave

# what one SDMA channel actually moves, to size the above against the hardware
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/11_flydsl_mega_moe_sdma/sdma_bandwidth.py
```

It is worth it only at large per-rank batches. `stage2+combine` against aiter,
eager, fp8 transport, `--chunk-tiles 16`, glm52 on 8x MI355, matching aiter's
output exactly (relL2 0) at every size (`--no-aligned-pair` for 8192 and up):

| tokens/rank | 128 | 512 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|
| vs aiter | 0.71x | 0.78x | 1.01x | **1.12x** | **1.23x** | **1.35x** |

Below ~2048 the fixed costs of the extra plan kernel and of the combine outweigh
what the copy engine saves, and aiter's P2P scatter wins.

The 8-wave GEMM2 against the default one, same harness (aiter `f0b96f49e`, mori
`c7da62db`, eager, `--chunk-tiles 16`, `--no-aligned-pair`, bit-exact at every size):

| tokens/rank | 512 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| v4_pro, `--gemm2 aiter` | 0.82x | 0.98x | 1.08x | 1.19x |
| v4_pro, `--gemm2 8wave` | 0.71x | **1.28x** | **1.40x** | **1.59x** |
| glm52, `--gemm2 aiter` | 0.75x | 1.00x | 1.26x | 1.37x |
| glm52, `--gemm2 8wave` | 0.77x | **1.17x** | **1.40x** | **1.39x** |

It helps most where GEMM2 is largest (v4_pro, `inter_dim` 3072); the same kernel
is aiter's opt-in `MegaMoEV2(stage2_sdma=True)` path (ROCm/aiter#7196).

The send/landing layout is exact: the plan kernel reads aiter's all-gathered
route counts, so every rank's run starts and lands at its real offset and nothing
is reserved per rank pair. The landing buffer holds at most a rank's own routes and
never overflows; the send buffer is `--send-slack` (default 4) times the even
share, capped by the 32-bit buffer-resource range. With fp8 transport a row past
it is not dropped: the plan marks it and GEMM2 writes it into the origin rank's
`comb_inp` in aiter's own P2P layout, so skewed routing stays exact. That path is
slow, but only a rank receiving more than `--send-slack` times the average takes
it: with `test --hot-bias 1.0` (v4_pro, bs=4096) stage2+combine is 5.3 ms
(`--gemm2 aiter`) / 4.6 ms (`8wave`) against aiter's 4.35 ms, where a per-pair
reservation took 15 ms. With bf16 transport (`--quant none`) such rows are still
dropped and `test` fails.

Example 12 also uses `torchrun` and SDMA. Its start kernel reads each peer's READY
once and posts a plain put when it is already set, `post_wait_mem_put_ptr` (the
copy engine waits) when it is not; `--mode cu` spins a CU on READY instead, and
`--skew-us` makes the last rank late:

```bash
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/12_flydsl_sdma_allgather_engine_wait/main.py --skew-us 200
```

Example 13 also uses `torchrun` and SDMA, and needs aiter (`MegaMoEV2` with the compact
prepare; measured with aiter `f0b96f49e` plus the files of ROCm/aiter#7196) and FlyDSL 0.4.1,
on a gfx950 node with 8 GPUs; mori must be built with `BUILD_CCO_SDMA=ON`. It replaces only
the Stage1 launch of aiter's operator (prepare, Stage2 and combine stay aiter's), so the layer
output must match aiter's byte for byte; `test` checks that over 4 back-to-back calls with
alternating inputs, `bench` times Stage1 (including prepare) and the whole layer with CUDA Graphs:

```bash
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/13_flydsl_mega_moe_stage1_sdma/main.py test --bs 8192
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 \
  examples/cco/python/13_flydsl_mega_moe_stage1_sdma/main.py bench --bs-list 2048,4096,8192
```

`--scale-mode` picks how GEMM1 gets the A scales of the landed rows. `shuffler`: a few
workgroups (`--shufflers`, default 4) put them into the contiguous GEMM's layout once per
M-tile, and a work item waits only on their flags. `ldsw`: every work item waits on the 7 peer
signals and stages its M-tile's scales into LDS. `auto` (default) takes the shuffler when
aiter's Stage1 `sort_block_m` is 128 (the 4096 and 8192 buckets) and ldsw otherwise. Stage1
including prepare against aiter, v4_pro on 8x MI355X, CUDA Graph, max over ranks, byte-exact:

| tokens/rank | 2048 | 4096 | 8192 |
|---|---|---|---|
| `shuffler` (2 or 4 blocks) | +0.07..+0.09 ms | -0.04..-0.05 ms | **-0.17..-0.23 ms** (aiter 2.40..2.43) |
| `ldsw` | +0.008 ms | +0.005 ms | -0.07 ms |

The shuffler delays the first M-tile by expert 0's shuffle, which only pays off at large
batches. The whole-layer gain is smaller than the Stage1 gain (8192: -0.09..-0.12 ms).
`main.py bench` with the default `auto` on one node (ms):

| tokens/rank | scale mode | Stage1 aiter | Stage1 SDMA | layer aiter | layer SDMA |
|---:|---|---:|---:|---:|---:|
| 2048 | ldsw | 0.933 | 0.958 (+0.025) | 1.572 | 1.622 (+0.050) |
| 4096 | shuffler | 1.439 | 1.369 (-0.070) | 2.466 | 2.443 (-0.023) |
| 8192 | shuffler | 2.431 | 2.212 (**-0.219**) | 4.548 | 4.422 (-0.126) |

The aiter baseline above is main `f0b96f49e`. Against main plus the open ROCm/aiter#7001
(faster compact Stage1 producer and GEMM1 tiles; same node, interleaved), aiter's Stage1 is
0.820 / 1.177 / 2.040 ms at 2048 / 4096 / 8192 and this example is **slower** by 0.13 / 0.19 /
0.18 ms (layer +0.17 / +0.20..+0.26 / +0.21 ms): once the CU dispatch is that cheap, the
extra pack copy (about 0.19 ms at 8192) costs more than the CU time the copy engines save.

`--gemm1 7001` (needs aiter with the files of ROCm/aiter#7001) runs #7001's own Stage1 GEMM1
tiles instead of the 8-wave GEMM1, in one launch that also packs the rows: producer
workgroups pack each (destination, segment) in expert order and the SDMA puts of a segment
start as soon as it is packed, so no pack launch runs before it. Against aiter + #7001 on one
node (CUDA Graph, Stage1 including prepare, ms):

| tokens/rank | aiter + #7001 | `--fuse-quant` (8-wave) | `--gemm1 7001` |
|---|---|---|---|
| 2048 | 0.813-0.820 | 0.911-0.915 | 0.872-0.874 |
| 4096 | 1.167-1.174 | 1.333-1.335 | 1.171-1.173 |
| 8192 | 2.022-2.052 | 2.142-2.149 | 2.010-2.014 |

Even at 4096 and 8192, still about 0.05 ms slower at 2048. The copy engines bound it now: KFD
recommends one XGMI SDMA engine per peer link on this node, and the transfer alone takes
about as long as #7001's GEMM1 without its dispatch.

`MORI_CCO_GDA_CONN=full` is required for GDA (03, 06) when both ranks share one
node; LSA examples (02, 04, 05) ignore it. Each example prints `SUCCESS` on pass.

---

## 3. Run the Triton example

The Triton integration links `libmori_cco_device.bc` with code object version 5
and passes `DevCommHandle.ptr` / `RegisteredWindow.handle` into the kernel.

```bash
# LSA
torchrun --standalone --nproc_per_node=2 \
  examples/cco/ir/test_triton_cco.py --transport lsa

# SDMA: MORI must also be built with BUILD_CCO_SDMA=ON
MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=2 \
  examples/cco/ir/test_triton_cco.py --transport sdma

# GDA-FULL on a single Ionic rail
MORI_DEVICE_NIC=ionic MORI_DISABLE_TOPO=1 MORI_RDMA_DEVICES=rocep9s0 \
  torchrun --standalone --nproc_per_node=2 \
  examples/cco/ir/test_triton_cco.py --transport gda
```

Replace `rocep9s0` with one active HCA on the host. Pinning one HCA avoids the
known local cross-rail hang on multi-AINIC systems; it is a correctness
workaround and limits aggregate bandwidth.

For latency/bandwidth sweeps and automated comparison with the C++ binaries,
use `benchmark/cco/triton/bench_p2p.py` and
`benchmark/cco/compare_triton_cpp.py`; see `benchmark/cco/README.md`.

---

## 4. Run the C++ examples

Two ways:

**(a) Build + install with the package.** `BUILD_EXAMPLES=ON` ships the binaries
into `site-packages/mori/examples/cco/`:

```bash
BUILD_EXAMPLES=ON pip install .
SP=$(python3 -c 'import mori, os; print(os.path.dirname(mori.__file__))')
export MORI_SOCKET_IFNAME=<iface>
mpirun --allow-run-as-root -n 2 $SP/examples/cco/cco_lsa_put
mpirun --allow-run-as-root -n 2 $SP/examples/cco/cco_gda_put
```

**(b) Build in a local `build/` and run in place** (dev loop):

```bash
cd <repo>
pip install pybind11 -q
cmake -S . -B build -GNinja -DBUILD_EXAMPLES=ON -DGPU_TARGETS=gfx942
ninja -C build cco_lsa_put cco_gda_put
export MORI_SOCKET_IFNAME=<iface>
mpirun --allow-run-as-root -n 2 ./build/examples/cco_lsa_put     # no LD_LIBRARY_PATH needed
mpirun --allow-run-as-root -n 2 ./build/examples/cco_gda_put
```

The example binaries carry an `$ORIGIN/../..` rpath (for the installed location)
plus the build-tree rpath, so they find `libmori_*.so` either way.

For two physical nodes (real cross-node GDA), launch one rank per node with
`MORI_CCO_GDA_CONN=crossnode`; rank 0 generates the cco `UniqueId` and shares it
with the other rank out-of-band (MPI bcast, or write it to a file the other rank
reads — see each example's bootstrap docstring).

---

All examples are single-node, 2-rank by default and print `SUCCESS` (the C++
ones also print `... put verified ...`). They are NIC-agnostic — the
`deploy-mori` skill installs the matching NIC userspace stack (AINIC / ConnectX /
Thor2-BNXT) for your host.
