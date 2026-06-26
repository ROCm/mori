# CCO device API → FlyDSL integration

Lets FlyDSL `@flyc.kernel` code call the cco GDA/LSA device API via a device-IR
binding layer (extern-C wrappers compiled to bitcode, linked into the kernel),
adapted to FlyDSL's scalar-only FFI. Fully independent of the shmem `mori.ir`
machinery; the C++ device implementation (`ccoGda<P>`) is reused **unchanged**.

## The 5 layers (top → bottom)

```
FlyDSL @flyc.kernel
  └─ cco.DevComm(h).gda(0).put(...)         OO handles (method-carrying fx.struct)
       └─ _bindings.PUT["iw__inc"](...)      per-combo FFI symbol tables
            └─ link_extern(ffi(...), bc)      lazy bind + attach bitcode
                 └─ llvm.call @cco_gda_put__iw__inc   FlyDSL-emitted IR (one monomorphic symbol)
                      └─ libmori_cco_device.bc extern "C" wrapper (one template instantiation)
                           └─ ccoGda<P>::put<Aggregate?,Coop>() the real C++ device impl (unchanged)
```

## What to look at where

| Question | File |
|---|---|
| What device ops exist / the C++ wrapper | `src/cco/device/cco_device_wrapper.cpp` |
| The ABI (exact symbol + scalar arg list) | `python/mori/cco/device/flydsl/_bindings.py` (1:1 with the wrapper) |
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

## Adding a new device op (the path)

1. Add the `extern "C"` wrapper in `cco_device_wrapper.cpp`. For a templated op,
   define it with the `CCO_TC_LIST` / `CCO_COOP_LIST` X-macros so every valid
   combination is monomorphized into a tagged symbol (no runtime dispatch). JIT
   re-compiles automatically — the cache key includes the wrapper source hash.
2. Add the matching symbol table / `_ffi(...)` prototype in `_bindings.py`
   (keyed by the same tags).
3. Expose it as a method on `DevComm` / `Window` / `Gda` in `handles.py`, picking
   the symbol from the compile-time `coop`/`thread_mode`/`signal_op` constants.

The `_bindings.py` ↔ wrapper ABI is mechanical — it's the natural codegen target.

## Examples

| Example | Shows |
|---|---|
| `03_flydsl_put` | GDA put + signal/wait (single-node FULL + 2-node CROSSNODE) |
| `04_flydsl_lsa_put` | LSA direct peer-pointer store in the kernel |
| `05_flydsl_lsa_allreduce` | LSA custom all-reduce: peer pointers + device signal barrier |
| `06_flydsl_gda_modes` | GDA template matrix: (thread_mode,coop) {indep×thread/warp/block, aggr×thread} × signal {inc,add} |

## FlyDSL kernel-author gotchas

1. Handle args on the `@flyc.jit` launcher must be typed `fx.Int64`, else the
   64-bit pointer is truncated → GPU fault.
2. A multi-field `fx.struct` can't be scf.if carried state ("state variable is
   list"); keep handles single-IR-value (extra fields → `fx.Constexpr`).
3. Don't name a `@flyc.jit` launcher `launch` (collides with `.launch()` in
   co_names → cache-key self-recursion).
4. A constant-count inner loop is lowered to dynamic `scf.for`; use
   `range_constexpr(N)` to unroll (e.g. peer accumulation in the all-reduce).
