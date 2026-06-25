# CCO device API → FlyDSL integration

Lets FlyDSL `@flyc.kernel` code call the cco GDA/LSA device API via a device-IR
binding layer (extern-C wrappers compiled to bitcode, linked into the kernel),
adapted to FlyDSL's scalar-only FFI. Fully independent of the shmem `mori.ir`
machinery; the C++ device implementation (`ccoGda<P>`) is reused **unchanged**.

## The 5 layers (top → bottom)

```
FlyDSL @flyc.kernel
  └─ cco.DevComm(h).gda(0).put(...)         OO handles (method-carrying fx.struct)
       └─ _bindings.cco_gda_put(...)         1:1 FFI prototypes
            └─ link_extern(ffi(...), bc)      lazy bind + attach bitcode
                 └─ llvm.call @cco_gda_put    FlyDSL-emitted IR
                      └─ libmori_cco_device.bc extern "C" wrapper (runtime template dispatch)
                           └─ ccoGda<P>::put() the real C++ device impl (unchanged)
```

## What to look at where

| Question | File |
|---|---|
| What device ops exist / the C++ wrapper | `src/cco/device/cco_device_wrapper.cpp` |
| The ABI (exact symbol + scalar arg list) | `python/mori/cco/device/flydsl/_bindings.py` (1:1 with the wrapper) |
| OO API (`DevComm/Window/Gda`) + enums (`CoopScope/SignalOp`) | `python/mori/cco/device/flydsl/handles.py` |
| `_ffi` factory + `cco_struct` (handle keeps methods across `if`/`for`) | `flydsl/_internal.py` |
| How the `.bc` is located at runtime | `python/mori/cco/device/bitcode.py` (`find_cco_bitcode`) |
| How the `.bc` is built (JIT, per arch+NIC+cov) | `bitcode.py::_jit_compile` (reuses `mori.jit`) |
| How to prebuild the `.bc` manually | `tools/build_cco_bitcode.sh` |
| 2-node launcher | `tools/run_cco_flydsl_2node.sh` |
| Runnable examples | `examples/cco/03..06_flydsl_*` |

## Key design points

- **Scalar handles.** FlyDSL FFI is scalar-only, so device objects cross as
  `uint64` intptrs (`ccoDevComm*`, `ccoWindow_t`); coop-scope / signal-op cross
  as `int32` enums. The wrapper reinterpret-casts and dispatches at runtime
  (`CCO_BY_COOP` + signalOp branch). Struct layout lives only in C++.
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

```bash
# 1. cco host extension must be built (PR #422); see python/mori/cco/

# 2. run an example (one node, 2 ranks). The device bitcode is JIT-compiled on
#    first use (cached in ~/.mori/jit, per arch+NIC+cov) — no manual build step.
export PYTHONPATH=python MORI_SOCKET_IFNAME=lo
export LD_LIBRARY_PATH=$(find build*/src -name '*.so' -printf '%h\n'|sort -u|tr '\n' ':')
mpirun -np 2 python examples/cco/03_flydsl_put/main.py     # GDA: set MORI_CCO_GDA_CONN=full intra-node

# (optional) prebuild the bitcode instead of JIT, or override the path:
#   bash tools/build_cco_bitcode.sh         # -> lib/libmori_cco_device.bc
#   export MORI_CCO_BC=/path/to/libmori_cco_device.bc

# 4. two physical nodes (GDA, CROSSNODE)
bash tools/run_cco_flydsl_2node.sh examples/cco/03_flydsl_put/main.py
```

## Adding a new device op (the path)

1. Add the `extern "C"` wrapper (use `CCO_DEV`, `CCO_BY_COOP` for coop dispatch)
   in `cco_device_wrapper.cpp`. (JIT re-compiles automatically — the cache key
   includes the wrapper source hash.)
2. Add the 1:1 `_ffi(...)` prototype in `_bindings.py` (matching scalar args).
3. Expose it as a method on `DevComm` / `Window` / `Gda` in `handles.py`.

The `_bindings.py` ↔ wrapper ABI is mechanical — it's the natural codegen target.

## Examples

| Example | Shows |
|---|---|
| `03_flydsl_put` | GDA put + signal/wait (single-node FULL + 2-node CROSSNODE) |
| `04_flydsl_lsa_put` | LSA direct peer-pointer store in the kernel |
| `05_flydsl_lsa_allreduce` | LSA custom all-reduce: peer pointers + device signal barrier |
| `06_flydsl_gda_modes` | GDA template matrix: coop {thread,warp,block} × signal {inc,add} |

## FlyDSL kernel-author gotchas

1. Handle args on the `@flyc.jit` launcher must be typed `fx.Int64`, else the
   64-bit pointer is truncated → GPU fault.
2. A multi-field `fx.struct` can't be scf.if carried state ("state variable is
   list"); keep handles single-IR-value (extra fields → `fx.Constexpr`).
3. Don't name a `@flyc.jit` launcher `launch` (collides with `.launch()` in
   co_names → cache-key self-recursion).
4. A constant-count inner loop is lowered to dynamic `scf.for`; use
   `range_constexpr(N)` to unroll (e.g. peer accumulation in the all-reduce).

## Known issue (not this integration)

cco host `ccoCommDestroy` aborts at `src/cco/cco_init.cpp:372`
(`hipMemAddressFree`) whenever windows were allocated — reproduces with pure
host code, no FlyDSL. It fires *after* data verification, so it only eats the
final "SUCCESS" print. Report to PR #422.
