# CCO device API → FlyDSL and Triton integration

Lets FlyDSL and Triton kernels call the CCO GDA/LSA/SDMA device API through
extern-C wrappers compiled to bitcode and linked into the kernel. The C++
device implementation (`ccoGda<P>` / `ccoSdma`) is reused unchanged.

## The layers (top → bottom)

```
CCO_DEVICE_FUNCTIONS (one scalar ABI table)
  ├─ FlyDSL: cco.DevComm(h).gda(0).put(...)
  │    └─ link_extern(ffi(...), cov6 bitcode)
  └─ Triton: mori.ir.triton.cco.gda_put_iw_inc(...)
       └─ core.extern_elementwise(..., cov5 bitcode)
            └─ llvm.call @cco_gda_put__iw__inc
                 └─ ccoGda<P>::put<...>() / ccoSdma::put<...>()
```

## What to look at where

| Question | File |
|---|---|
| What device ops exist / the C++ wrapper | `src/cco/device/cco_device_wrapper.cpp` |
| The shared ABI metadata | `python/mori/cco/device/ops.py` (`CCO_DEVICE_FUNCTIONS`) |
| FlyDSL FFI objects | `python/mori/cco/device/flydsl/_bindings.py` |
| Triton extern wrappers | `python/mori/ir/triton/cco/ops.py` |
| OO API (`DevComm/Window/Gda`) + enums (`CoopScope/SignalOp/ThreadMode`) | `python/mori/cco/device/flydsl/handles.py` |
| `_ffi` factory + `cco_struct` (handle keeps methods across `if`/`for`) | `flydsl/_internal.py` |
| How the `.bc` is located at runtime | `python/mori/cco/device/bitcode.py` (`find_cco_bitcode`) |
| How the `.bc` is built (JIT, per arch+NIC+cov) | `bitcode.py::_jit_compile` (reuses `mori.jit`) |
| How to prebuild the `.bc` manually | `tools/build_cco_bitcode.sh` |
| Runnable examples | `examples/cco/python/03..06_flydsl_*` |

## Key design points

- **Scalar handles.** FlyDSL FFI is scalar-only, so device objects cross as
  `uint64` intptrs (`ccoDevComm*`, `ccoWindow_t`); signal id/value cross as
  `int32`/`uint64`. Struct layout lives only in C++.
- **No dispatch overhead (monomorphization).** Each template axis
  (`Coop` × `ThreadMode` × `RemoteAction`) is compiled into a *distinct*
  `extern "C"` symbol, name-mangled with a tag (`cco_gda_put__iw__inc` = indep
  warp + signal-inc). `coop`/`thread_mode`/`signal_op` are compile-time constants
  (Python enums), so `handles.py` picks the symbol by name at trace time and the
  kernel emits **one direct call to one instantiation** — no runtime branch, no
  type erasure, nothing to constant-fold. Passing a runtime DSL value for any
  axis is a `TypeError` (it can't select a symbol). `always_inline` (`CCO_DEV`)
  then inlines the thin forwarder into the kernel.
- **ThreadMode is selectable** (`ThreadMode.INDEPENDENT` / `AGGREGATE`); the
  data path carries it as part of its `(ThreadMode,Coop)` tag (`it/iw/ib/at`).
  `AGGREGATE` is only valid with `CoopScope.THREAD` (cco coalesces the warp's
  lanes itself — enforced in both the wrapper and `handles.py`).
- **Concrete status returns.** Side-effect wrappers return `int32` status 0.
  FlyDSL ignores it; Triton needs a concrete element return type for
  `extern_elementwise`, especially when an argument is a block value.
- **Provider fixed at build time** via `CCO_GDA_BUILD_PROVIDER` (NIC macro) — one
  `ccoGda<P>` specialization per NIC, same as the C++ kernels / shmem bitcode.
- **OO handles** (`DevComm/Window/Gda`) are method-carrying `fx.struct`s
  (`cco_struct`): build once, reuse across control flow. `Gda.ctx` is `Constexpr`
  so the handle carries a single IR value.

## LSA vs GDA — different models

- **LSA** (intra-node P2P): cco only exposes `cco_lsa_ptr` → the peer's
  load/store-accessible VA. **The kernel operates on that pointer directly**
  (`buffer_load`/`buffer_store`); cco does NOT move the data. See examples 04/05.
- **GDA** (RDMA): opaque network ops, so they ARE exposed as device ops —
  `gda.put / put_value / get / signal / wait_signal / flush`. See examples 03/06.
- **SDMA** (intra-node copy engine): `sdma.put / get` (window + offset) and
  `put_ptr / get_ptr` (resolved addresses) mirror `ccoSdma`'s, with its template
  arguments as compile-time keywords (`coop`, `local_signal`, `remote_signal`,
  `aggregate` + `commit`, `signal_per_copy`, `same_queue`). A signal bumps
  `signal_slot(<sender lsa rank>, qid)` -- ours for `local_signal`, the peer's
  for `remote_signal` -- which `wait_signal` polls; `quiet / quiet_queue` drain
  regardless. See example 07.

## Build & run

Set up the environment with the `deploy-mori` skill (`.claude/skills/deploy-mori`),
then install MORI and the example runtime deps:

```bash
pip install .                          # builds + co-locates all libmori_*.so
pip install mpi4py "flydsl==0.2.2"     # mpi4py: bootstrap; flydsl: the device bindings
```

After `pip install .` no `PYTHONPATH` / `LD_LIBRARY_PATH` / `MORI_CCO_BC` is
needed — the libs are co-located in `site-packages/mori/` (RUNPATH `$ORIGIN`) and
the device bitcode is JIT-compiled on first use (cached in `~/.mori/jit`, per
arch+NIC+cov). The only run-time env var is the RDMA interface:

```bash
export MORI_SOCKET_IFNAME=<iface>      # e.g. enp159s0np0
export MORI_CCO_GDA_CONN=full          # required for GDA on a single node
mpirun -n 2 python examples/cco/python/03_flydsl_put/main.py
```

See [`examples/cco/README.md`](../../../../examples/cco/README.md) for the full
run guide (Python + C++), including the two-node (`crossnode`) setup. To skip
JIT, override with `MORI_CCO_BC=/path/to/libmori_cco_device.bc` or prebuild via
`tools/build_cco_bitcode.sh`.

Triton uses code object version 5 and passes CCO state explicitly:

```python
from mori.ir.triton import cco

kernel[(1,)](
    dc.ptr,
    win.handle,
    extern_libs=cco.get_extern_libs(),
)
```

No shmem `install_hook()` is used. Run the complete example with:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/cco/ir/test_triton_cco.py --transport lsa
```

## Adding a new device op (the path)

1. Add the `extern "C"` wrapper in `cco_device_wrapper.cpp`, as a call to the
   public device API (`ccoGda` / `ccoSdma` / ...), not its internals. Template
   axes cross the scalar FFI one of two ways, both with no runtime cost:
   - a tag in the symbol name (`cco_<op>__<tag>`), one symbol per value: the GDA
     ops via the `CCO_TC_LIST` / `CCO_COOP_LIST` X-macros, the SDMA coop scope;
   - a `flags` argument the wrapper dispatches on (`SdmaWithFlags` for
     `cco_sdma_{put,get}_ptr__<coop>`). The wrapper is inlined, so a
     compile-time constant `flags` folds to one instantiation; the Python
     handles always pass one. Prefer this for an axis with many combinations.
   SDMA wrappers go inside `#if BUILD_CCO_SDMA`. JIT re-compiles automatically —
   the cache key includes the wrapper source and the cco headers.
2. Add the scalar signature once to `CCO_DEVICE_FUNCTIONS` in `ops.py`. Give an
   SDMA op a `family` starting with `sdma_`: the bitcode tests use it to skip
   the op when BUILD_CCO_SDMA is off (the `#if` above is what drops the symbol).
   Triton exports the entry and FlyDSL binds it (`_bindings.ALL`) automatically.
3. Expose it as a method in both `flydsl/handles.py` and
   `mori/ir/triton/cco/handles.py`, with the keywords of the C++ template
   arguments, and add the method name to
   `test_triton_facades_match_flydsl_handle_names`.

`tests/python/cco/test_triton_bindings.py` checks that the cov5 bitcode defines
exactly the registered symbols (no more, no fewer) with the registered argument
and return types, and that FlyDSL binds the same set. It does not check step 3
beyond the method names.

## Examples

| Example | Shows |
|---|---|
| `03_flydsl_put` | GDA put + signal/wait (single-node FULL + 2-node CROSSNODE) |
| `04_flydsl_lsa_put` | LSA direct peer-pointer store in the kernel |
| `05_flydsl_lsa_allreduce` | LSA custom all-reduce: peer pointers + device signal barrier |
| `06_flydsl_gda_modes` | GDA template matrix: (thread_mode,coop) {indep×thread/warp/block, aggr×thread} × signal {inc,add} |
| `examples/cco/ir/test_triton_cco.py` | Triton DevComm queries plus LSA, SDMA, and GDA |

## FlyDSL kernel-author gotchas

1. Handle args on the `@flyc.jit` launcher must be typed `fx.Int64`, else the
   64-bit pointer is truncated → GPU fault.
2. A multi-field `fx.struct` can't be scf.if carried state ("state variable is
   list"); keep handles single-IR-value (extra fields → `fx.Constexpr`).
3. Don't name a `@flyc.jit` launcher `launch` (collides with `.launch()` in
   co_names → cache-key self-recursion).
4. A constant-count inner loop is lowered to dynamic `scf.for`; use
   `range_constexpr(N)` to unroll (e.g. peer accumulation in the all-reduce).
