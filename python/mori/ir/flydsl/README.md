# mori ↔ DSL integration contract (FlyDSL-first)

This document describes the **integration contract** between mori's shmem
device-side API and a host DSL/compiler (currently FlyDSL; the same
pattern applies to any DSL that can emit `llvm.call` against an external
bitcode library).  Read this if you are:

* a **user** writing `@flyc.kernel` code that calls mori shmem ops, or
* a **DSL author** wiring a new front-end on top of mori's
  `libmori_shmem_device.bc`.

For mori's framework-agnostic bitcode ABI and the full list of shmem
device functions, see [`../README.md`](../README.md).

## 1. For users: writing `@flyc.kernel` that uses mori shmem

### Recommended import form

```python
from mori.ir import flydsl as mori_shmem

@flyc.kernel
def my_kernel(buf: fx.Tensor):
    pe  = mori_shmem.my_pe()
    npe = mori_shmem.n_pes()
    mori_shmem.putmem_nbi_warp(buf, buf, 64, (pe + 1) % npe, 0)
    mori_shmem.quiet_thread_pe((pe + 1) % npe)
```

**Do not** use `from mori.ir.flydsl import *`.  The star-import iterates
`__all__`, materialising every `ExternFunction` and forcing the one-time
`libmori_shmem_device.bc` build (~15-20 s cold, micro-seconds warm) at
module-import time, defeating the lazy-init design.

### Cold-start cost & warm-up

`mori.ir.flydsl` is lazy: simply `import mori.ir.flydsl` costs nothing.
The first attribute access (`mori_shmem.my_pe` etc.) triggers the mori
shmem bitcode build.  To move that cost off the kernel-launch critical
path, call `prepare()` once at process start:

```python
from mori.ir import flydsl as mori_shmem
mori_shmem.prepare()           # optional; idempotent
```

This is **not** FlyDSL's per-kernel JIT cache.  FlyDSL's MLIR→LLVM JIT is
orthogonal and paid lazily on the first `@flyc.kernel` call; `prepare()`
only warms the **mori-side** bitcode that every such kernel will later
link against via `link_libs`.

Alternatively, warm the cache offline during image build / CI:

```bash
MORI_PRECOMPILE=1 python -c 'import mori'
```

## 2. For DSL authors: the integration surface

mori exposes exactly **three things** to any host DSL:

1. **Bitcode path.**  `mori.ir.bitcode.find_bitcode(cov=...)` returns the
   absolute path of `libmori_shmem_device.bc` for the requested code
   object version.  Your DSL's linker feeds this path to
   `rocdl-attach-target` (or an equivalent pass) so the shmem symbols
   get resolved at GPU-binary generation time.
2. **ABI metadata.**  `mori.ir.ops.MORI_DEVICE_FUNCTIONS` is a dict of
   `name → {symbol, args, ret, pure}` entries that your DSL wraps as
   callable objects (e.g. FlyDSL's `ExternFunction`).  The `pure`
   flag should be forwarded to the LLVM `readnone/willreturn`
   attributes once your DSL lowers extern calls with attribute support
   — mori only declares which symbols are pure, it does not enforce
   the attribute itself.
3. **Post-load module-init callable.**  `mori.shmem.shmem_module_init`
   is a top-level Python callable of signature `(hipModule_t) -> None`
   that writes device-side pointers into the loaded module's
   `__global__` state.  Your DSL's JIT runtime must invoke this **once
   per loaded `hipModule_t`**, before the first kernel in that module
   is launched.

The single mori-side source of truth for these three handles is
[`compile_helper.get_flydsl_compile_info()`](compile_helper.py), which
returns a `FlyDSLCompileInfo` dataclass.  Any future mori knob (per-arch
COV selection, NIC variants, …) goes through that function without
touching DSL code.

### How FlyDSL consumes the contract

[`ops.py`](ops.py) generates one `ExternFunction` wrapper per entry in
`MORI_DEVICE_FUNCTIONS`, baking `bitcode_path` and `module_init_fn`
into each wrapper.  When a `@flyc.kernel` body calls such a wrapper,
`ExternFunction._ensure_declared` populates the current FlyDSL
`CompilationContext`:

* `ctx.link_libs.add(bitcode_path)` — feeds the `rocdl-attach-target`
  link option list.
* `ctx.post_load_processors.append(module_init_fn)` — queued to run
  after `ExecutionEngine.initialize()` loads each GPU module.

No mori import ever happens on FlyDSL's JIT path; everything flows
through `CompilationContext`.  A new DSL backend can reproduce the same
wiring and still reuse mori's bitcode and module-init callable
unchanged.

### Implementing a new DSL backend

A compliant DSL backend must:

| Requirement | Why |
|---|---|
| Pass `bitcode_path` to its equivalent of `rocdl-attach-target` for every kernel that uses a mori shmem op. | Otherwise the GPU binary has unresolved externs. |
| Call `module_init_fn(hipModule_t)` exactly once after each module load, on the same process, before any kernel in that module runs. | Otherwise mori's device-side globals stay uninitialised → first shmem call faults. |
| Make the `module_init_fn` callable survive on-disk caching (top-level function, or an equivalent re-resolvable identifier). | Disk caches must not silently drop initialisers; see "Pickling contract" below. |

## 3. Pickling / on-disk JIT cache contract

FlyDSL's `CompiledArtifact` on-disk cache **does not pickle
`ExternFunction` instances** — they are module-level singletons and are
looked up fresh per process via normal `import`/attribute access.

What *does* get pickled is the list of `module_init_fn` callables in
`CompiledArtifact._post_load_processors`.  These are serialised as
`"module:qualname"` strings (see
`FlyDSL/python/flydsl/compiler/jit_executor.py::_qualname`) and
re-imported on cache hit.

**Constraint for DSL authors and users:** every `module_init_fn`
you register must be a **top-level callable** in an import-reachable
Python module.  Lambdas, `functools.partial`, and bound methods are
**rejected** at pickle time (a `pickle.PicklingError` is raised when
the cache tries to persist such an artifact).  This is by design — the
alternative (silently dropping processors) would let a cached kernel
round-trip and then GPU-fault on the next process with no stack clue
pointing back to the missing initialiser.

`mori.shmem.shmem_module_init` is a plain top-level function and
satisfies this constraint.

## 4. Files in this package

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | Lazy package entry + `prepare()` + `__getattr__` forwarding |
| [`ops.py`](ops.py) | `ExternFunction` wrappers generated from `MORI_DEVICE_FUNCTIONS` |
| [`compile_helper.py`](compile_helper.py) | Single source of truth for compile metadata (`get_flydsl_compile_info`) |

## 5. Related documents

* [Mori IR bitcode ABI](../README.md) — framework-agnostic symbol list and JIT build details.
* [FlyDSL extern-integration guide](https://github.com/ROCm/FlyDSL/blob/main/docs/extern_integration_guide.md) — FlyDSL-side view of the same contract: `ExternFunction`, `link_libs`, post-load callback, and the `mgpuSetModuleLoadCallback` C++ concurrency contract.
