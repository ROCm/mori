# CCO examples

Runnable examples for the **cco** GPU-communication API — both Python (FlyDSL +
the cco host runtime) and C++.

```
examples/cco/
├── python/   # FlyDSL device kernels + cco host runtime (mpi4py bootstrap)
└── cpp/      # standalone C++ host+device examples (MPI bootstrap)
```

| Example | Lang | Shows |
|---|---|---|
| `python/01_barrier` | py | cco host barrier across ranks |
| `python/02_lsa_put` | py | intra-node LSA put via a hand-written `.hip` kernel (mori.jit) |
| `python/03_flydsl_put` | py | FlyDSL GDA put + signal/wait |
| `python/04_flydsl_lsa_put` | py | FlyDSL LSA: direct peer-pointer store in the kernel |
| `python/05_flydsl_lsa_allreduce` | py | FlyDSL LSA custom all-reduce (peer pointers + device signal barrier) |
| `python/06_flydsl_gda_modes` | py | FlyDSL GDA template matrix: (thread_mode, coop) × signal |
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
pip install mpi4py "flydsl==0.2.2"
```

- `mpi4py` — every example bootstraps the cco `UniqueId` over MPI.
- `flydsl==0.2.2` — required by the Python **FlyDSL** examples (03–06). Pinned to
  the FlyDSL ABI the device bitcode targets. (Also available as the optional
  extra `pip install amd_mori[flydsl]`.) Not needed for `01`, `02`, or the C++
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

`MORI_CCO_GDA_CONN=full` is required for GDA (03, 06) when both ranks share one
node; LSA examples (02, 04, 05) ignore it. Each example prints `SUCCESS` on pass.

---

## 3. Run the C++ examples

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
