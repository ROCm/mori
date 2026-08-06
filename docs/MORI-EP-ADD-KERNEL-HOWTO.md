# How to add a new MoE dispatch/combine kernel to MORI and drive it from SGLang

Audience: someone who has an experimental dispatch and/or combine HIP kernel and wants
to A/B it against the shipped MORI kernels inside SGLang, without forking either repo's
call path.

**Recommended approach: add the kernel to MORI as a new `KernelType`, then select that
`KernelType` explicitly from SGLang via an env var.** Do *not* monkey-patch
`mori.ops.EpDispatchCombineOp` from SGLang, and do not replace the body of an existing
kernel — you lose the ability to compare against the baseline in the same binary.

Everything below assumes the shipped intra-node (P2P/XGMI) kernels as the template. The
inter-node ones follow the same wiring, they just allocate more buffers.

---

## 0. Mental model — what a "KernelType" actually is

A `KernelType` is a tuple of four things:

| Layer | What it holds | Where |
|---|---|---|
| Enum value | The selector, serialized into the packed config | `include/mori/ops/dispatch_combine/dispatch_combine.hpp:56` |
| Symmetric-memory buffer set | Which shmem buffers `EpDispatchCombineHandle` allocates | `src/ops/dispatch_combine/dispatch_combine.cpp` (`InitializeShmemBuf`) |
| One `.hsaco` compilation unit | The `.hip` TU that gets JIT-compiled and loaded | `src/ops/kernels/ep_*.hip` |
| A launch sequence | Which `extern "C"` symbols get launched, with what grid/block/smem | `python/mori/ops/dispatch_combine.py` (Python path) and `src/ops/dispatch_combine/launch.cpp` (C++ path) |

The kernel *bodies* are `__device__` function templates in
`src/ops/dispatch_combine/*.hpp`; the `.hip` TU instantiates them into `extern "C"
__global__` symbols named `<KernelName>_<dtype_suffix>` via the `WRAP_*` macros in
`src/ops/kernels/ep_common.hip`. The host side launches them **by string name** through
`hipModuleLaunchKernel` — there is no link-time coupling. That's why adding a kernel is
mostly bookkeeping, and why a typo in the symbol name shows up as a runtime
"Kernel function not found in any loaded module" rather than a link error.

Args are passed as one flat POD struct, `EpDispatchCombineArgsRaw` (host) /
`EpDispatchCombineArgs<T>` (device). There is a `static_assert` that the two layouts
match. **If you add a field, add it to both, in the same position.**

---

## 1. Write the kernel body

Create `src/ops/dispatch_combine/myexp.hpp` (pick a real name). Mirror
`intranode.hpp`:

```cpp
#pragma once
#include "mori/ops/dispatch_combine/dispatch_combine.hpp"
#include "src/ops/dispatch_combine/common.hpp"

namespace mori {
namespace moe {

// The `_body` suffix is a convention the WRAP_* macros depend on:
// WRAP(EpDispatchMyExpKernel, bf16, hip_bfloat16) emits a __global__ that calls
// EpDispatchMyExpKernel_body<hip_bfloat16>.
template <typename T>
__device__ void EpDispatchMyExpKernel_body(EpDispatchCombineArgs<T> args) {
  // ... your dispatch implementation ...
}

template <typename T, bool UseP2PRead = true>
__device__ void EpCombineMyExpKernel_body(EpDispatchCombineArgs<T> args) {
  // ... your combine implementation ...
}

}  // namespace moe
}  // namespace mori
```

Reference implementations, in increasing order of complexity:
`intranode_ll.hpp` (simplest), `intranode.hpp`, `internode_v1.cpp`,
`low_latency_async.cpp`.

Practical notes:

- Read peer memory through `SymmMemObjPtr` (e.g. `args.intraNodeTokBufs.dispatchOut`),
  not through raw pointers — that's what makes the kernel work under both the P2P and
  RDMA providers.
- Everything in this header is compiled by `hipcc` only. Don't include host RDMA headers.
- If you index by `warpSize / numExpertPerToken` or similar, check the case where topk
  does **not** divide the warp size. A real slot-assignment bug (double-allocated slots,
  silent corruption before an eventual assert) came from exactly that assumption with
  topk=6.

## 2. Create the `.hip` translation unit

Create `src/ops/kernels/ep_myexp.hip`:

```cpp
// KernelType::MyExp — experimental intra-node dispatch + combine.
#include "src/ops/kernels/ep_common.hip"
#include "src/ops/dispatch_combine/myexp.hpp"   // your header, after ep_common.hip

MORI_DEFINE_GPU_STATES   // required exactly once per .hsaco

WRAP_ALL_TYPES(EpDispatchMyExpKernel)                    // -> _bf16 / _f32 / _fp8_* / _fp4
WRAP_ALL_TYPES_BOOL(EpCombineMyExpKernel, _p2p,  true)
WRAP_ALL_TYPES_BOOL(EpCombineMyExpKernel, _nop2p, false)
```

Why a separate TU instead of appending to `ep_intranode.hip`: it keeps your compile time
and your blast radius to yourself, and it means the baseline `.hsaco` is byte-identical to
main while you iterate.

`MORI_DEFINE_GPU_STATES` defines the per-module `globalGpuStates` that
`shmem_module_init` fills in at load time. Omitting it produces a module that loads fine
and then faults on the first shmem access.

The macro zoo (`WRAP`, `WRAP_BOOL`…`WRAP_BOOL8`, `WRAP_ALL_TYPES*`) is in
`src/ops/kernels/ep_common.hip:69`. If your kernel takes more template bools than the
existing macros cover, add a `WRAP_BOOLn` next to them.

## 3. Add the enum value

`include/mori/ops/dispatch_combine/dispatch_combine.hpp:56`:

```cpp
enum KernelType {
  IntraNode = 0,
  InterNode = 1,
  InterNodeV1 = 2,
  InterNodeV1LL = 3,
  AsyncLL = 4,
  IntraNodeLL = 5,
  MyExp = 6,        // append — never renumber
};
```

**Append only.** The enum is serialized by value into `ToPackedI32Array()` (slot 14) and
read back by `FromPackedI32Array()`; renumbering silently reinterprets configs.

Then expose it to Python in `src/pybind/pybind_ops.cpp:360`:

```cpp
      .value("MyExp", mori::moe::KernelType::MyExp)
```

## 4. Wire up the host handle

The handle decides which symmetric buffers to allocate. Every place that currently
special-cases `KernelType::IntraNode` has to learn about `MyExp` — if you reuse the
intra-node buffer set, this is purely additive. The exhaustive list (on today's main):

```
include/mori/ops/dispatch_combine/dispatch_combine.hpp:295   GetShmemDispatchOutTokMemObj
include/mori/ops/dispatch_combine/dispatch_combine.hpp:303   GetShmemCombineOutTokMemObj
include/mori/ops/dispatch_combine/dispatch_combine.hpp:311   GetShmemCombineInpTokMemObj
src/ops/dispatch_combine/dispatch_combine.cpp:199            blockwise-quant staging sizing
src/ops/dispatch_combine/dispatch_combine.cpp:211            InitializeShmemBuf   (allocate)
src/ops/dispatch_combine/dispatch_combine.cpp:290            FinalizeShmemBuf     (free)
src/ops/dispatch_combine/dispatch_combine.cpp:484            GetEpDispatchCombineArgsRaw (pick variant)
src/ops/dispatch_combine/dispatch_combine.cpp:435            crossDeviceBarrierFlag init
src/ops/dispatch_combine/convert.hpp:187,222                 StdMoE convert (only if ENABLE_STANDARD_MOE_ADAPT)
```

Before you start, run this and mirror every hit:

```bash
grep -rn "KernelType::IntraNode\b" src/ include/ | grep -v _jit-sources
```

Missing one of these is the single most common way to get a plausible-looking kernel that
reads uninitialized `SymmMemObjPtr`s. In particular `GetEpDispatchCombineArgsRaw` uses
`std::get<ShmemBufsIntraNode>(shmemTokBufs)` — a `std::variant` — so a missed branch is a
`std::bad_variant_access` throw, not a crash. Take the throw as the signal.

If your kernel needs a genuinely different buffer set, add a `ShmemBufsMyExp` struct next
to `ShmemBufsIntraNode` (`dispatch_combine.hpp:217`), add it to the `std::variant` at
`dispatch_combine.hpp:356` and to the mirrored field in both args structs.

## 5. Wire up the launch path (Python — this is the one SGLang uses)

`python/mori/ops/dispatch_combine.py`:

**a. Map the kernel type to its `.hsaco` TU** (`:189`):

```python
_KERNEL_TYPE_TO_HIP = {
    ...
    EpDispatchCombineKernelType.MyExp: "ep_myexp",   # = src/ops/kernels/ep_myexp.hip
}
```

**b. Add the dispatch branch** in `EpDispatchCombineOp.dispatch()` (`:743` is the
IntraNode branch — copy it):

```python
        elif kt == EpDispatchCombineKernelType.MyExp.value:
            self._launch(
                f"EpDispatchMyExpKernel_{sfx}", grid, block, shared_mem, stream, args_ptr
            )
```

`sfx` comes from `_DTYPE_SUFFIX[input.dtype]` and must match the suffix your `WRAP_*`
macro emitted. Multi-kernel sequences use `self._launch_multi([...names], [grids],
[blocks], [smems], stream, args_ptr)` — see the `AsyncLL` branch at `:761`.

**c. Add the combine branch** at `:1063` (the IntraNode/IntraNodeLL branch). Note it
picks `_nop2p` vs `_p2p` from `use_external_inp_buf`; keep that convention so the
existing `--zero-copy` knob keeps meaning the same thing.

**d.** If your kernel needs a split send/recv (like `AsyncLL`), also extend
`dispatch_recv()` (`:846`) and `combine_recv()`.

## 6. Optional: the C++ launch path

`src/ops/dispatch_combine/launch.cpp` (`LaunchDispatch` `:447`, `LaunchCombine` `:529`) is
a separate, Python-free launcher used by the C++ tests/examples and the AOT `.hsaco`
loader. SGLang does not use it. Add a `case KernelType::MyExp:` there if you want
`examples/ops/dispatch_combine/test_dispatch_combine.cpp` to work; skip it otherwise — but
if you do add it, keep the kernel-name selection logic identical to the Python side. The
two paths have drifted before, which is why several comments in both files say "keep in
sync".

## 7. Build / install

The kernels are **JIT-compiled at runtime** (`hipcc --genco` → `.hsaco`, cached under
`~/.mori/jit/<arch>_<nic>/<hash>/`). Only steps 3, 4, 6 (C++/pybind) need a rebuild:

```bash
git config --global --add safe.directory /apps/ditian12/mori   # fresh containers only
cd /apps/ditian12/mori && pip install -e .
```

Two gotchas that cost people hours:

1. **Editable vs wheel install.** `get_mori_source_root()`
   (`python/mori/jit/config.py:295`) prefers the repo root, and falls back to the
   *packaged copy* at `<site-packages>/mori/_jit-sources/`. Inside a prebuilt SGLang
   image, mori is usually a wheel install — so editing `src/ops/kernels/ep_myexp.hip` in
   your checkout does **nothing**. Either `pip install -e .` from the checkout, or edit
   `_jit-sources/` too (`setup.py:317 _copy_jit_sources` is what populates it).

2. **JIT cache invalidation.** The cache key hashes the whole subsystem source tree
   (`src/ops/**` + `include/mori/**`), not just the top-level `.hip`, so edits to your
   header do trigger a recompile. If you ever suspect a stale `.hsaco`, `rm -rf
   ~/.mori/jit` — that is always safe.

Pre-compile before forking workers (avoids N processes racing the same `hipcc`):

```bash
MORI_PRECOMPILE=1 python -c "import mori"
# or, in-process, before spawn:
mori.ops.dispatch_combine.warmup_jit_kernels(mori.ops.EpDispatchCombineKernelType.MyExp)
```

## 8. Validate standalone, before touching SGLang

Do not debug a new kernel through a 700B model. Copy
`tests/python/ops/test_dispatch_combine_intranode.py`, swap `kernel_type=` (`:56`, `:111`),
and run it against the reference implementation in `dispatch_combine_test_utils.py`. Both
harnesses spawn their own ranks — no `torchrun`:

```bash
cd /apps/ditian12/mori
pytest -sv tests/python/ops/test_dispatch_combine_intranode.py       # correctness
python3 tests/python/ops/bench_dispatch_combine.py \
        --max-tokens 4096 --dtype bf16 --world-size 8                # perf
```

The bench builds its `EpDispatchCombineConfig` at `bench_dispatch_combine.py:926` without
passing `kernel_type`, i.e. it always benches `IntraNode`. Add `kernel_type=...` there (or
a `--kernel-type` arg) before trusting its numbers for your kernel.

Profiling: build with `ENABLE_PROFILER=ON` and use mori's built-in profiler
(`--cmd profile`), not `rocprofv3` — the built-in one has the per-phase markers these
kernels are instrumented with. See `docs/PROFILER.md`.

Only move on once correctness matches the baseline `KernelType` bit-for-bit (or within the
quant tolerance) at your target EP size and token counts.

---

## 9. Selecting the kernel from SGLang

All MORI EP integration lives in one file:
`python/sglang/srt/layers/moe/token_dispatcher/moriep.py`.

Today the kernel type is chosen **implicitly** from topology
(`init_mori_op`, `:253`):

```python
mode = EpMode.INTRA_NODE if world_size <= 8 else EpMode.INTER_NODE
if deepep_mode.enable_low_latency() or enable_sdma:
    mode = EpMode.LOW_LATENCY
cfg = get_ep_dispatch_configs(num_max_dispatch_tokens_per_rank)[mode]
```

and `get_ep_dispatch_configs()` (`:170`) maps each `EpMode` to an `EpDispatchConfig`
(kernel type + `warp_num_per_block` / `block_num` / `rdma_block_num`).

The cheapest A/B hook is an explicit env override, in the spirit of the existing
`SGLANG_MORI_DISPATCH_DTYPE` / `SGLANG_MORI_COMBINE_DTYPE` overrides. Add to `moriep.py`:

```python
_EXPLICIT_KERNEL_TYPES = {
    "intra_node":      ("IntraNode",      16, 80, 0),
    "intra_node_ll":   ("IntraNodeLL",    16, 80, 0),
    "inter_node_v1":   ("InterNodeV1",     8, 64, 32),
    "inter_node_v1ll": ("InterNodeV1LL",   8, 64, 32),
    "async_ll":        ("AsyncLL",         8, 64, 32),
    "myexp":           ("MyExp",          16, 80, 0),   # <-- new kernel
}


def _kernel_type_override():
    """SGLANG_MORI_KERNEL_TYPE=myexp forces a specific mori KernelType.

    Returns an EpDispatchConfig, or None to keep the topology-derived default.
    """
    import mori

    name = os.environ.get("SGLANG_MORI_KERNEL_TYPE", "").strip().lower()
    if not name or name == "auto":
        return None
    if name not in _EXPLICIT_KERNEL_TYPES:
        raise ValueError(
            f"SGLANG_MORI_KERNEL_TYPE={name!r} not in {sorted(_EXPLICIT_KERNEL_TYPES)}"
        )
    attr, wpb, bn, rbn = _EXPLICIT_KERNEL_TYPES[name]
    kt = getattr(mori.ops.EpDispatchCombineKernelType, attr, None)
    if kt is None:
        raise RuntimeError(
            f"installed mori has no KernelType.{attr}; rebuild mori with the new kernel"
        )
    return EpDispatchConfig(
        kernel_type=kt,
        warp_num_per_block=get_int_env_var("SGLANG_MORI_WARP_NUM_PER_BLOCK", wpb),
        block_num=get_int_env_var("SGLANG_MORI_BLOCK_NUM", bn),
        rdma_block_num=get_int_env_var("SGLANG_MORI_RDMA_BLOCK_NUM", rbn),
    )
```

and in `init_mori_op`, right after the existing `cfg = get_ep_dispatch_configs(...)[mode]`:

```python
    cfg = _kernel_type_override() or cfg
```

Then run stock SGLang with:

```bash
SGLANG_MORI_KERNEL_TYPE=myexp   # baseline: unset, or =intra_node
```

Three things to check on the SGLang side:

- **`getattr(..., attr, None)` is deliberate here.** It's the one place the codebase's
  "no defensive getattr" rule should bend: the point is to run the *same* SGLang against a
  mori build that may or may not have the new kernel, and fail with a clear message
  instead of `AttributeError` at model-init time. `init_mori_op` already does the same
  kind of version probing in `check_mori_compatibility()` (`:311`).
- **`init_mori_op` is `@lru_cache`d** (`:211`) on its arguments, and the env var is not one
  of them. That is fine for a process-lifetime env var; it means you cannot flip kernels
  mid-process. Restart the engine between A/B runs.
- **Async (send/recv-split) kernels take a different code path.** `moriep.py:836` and
  `:915` test `kernel_type is AsyncLL` to decide whether to use `dispatch_send` /
  `dispatch_recv`. If your kernel is a plain single-shot dispatch/combine (the
  intra-node shape), you need no change there. If it's send/recv-split, extend those two
  checks to a set membership rather than adding a second `is` comparison.

Nothing else in SGLang needs to change: `dispatch()` / `combine()` return the same
tensors regardless of kernel type.

---

## 10. Checklist

MORI:

- [ ] `src/ops/dispatch_combine/myexp.hpp` — `_body` templates
- [ ] `src/ops/kernels/ep_myexp.hip` — `MORI_DEFINE_GPU_STATES` + `WRAP_*`
- [ ] `KernelType::MyExp` appended (not renumbered) in `dispatch_combine.hpp:56`
- [ ] `.value("MyExp", ...)` in `pybind_ops.cpp:360`
- [ ] every `KernelType::IntraNode` hit in `src/ include/` mirrored
- [ ] `_KERNEL_TYPE_TO_HIP` entry in `python/mori/ops/dispatch_combine.py:189`
- [ ] dispatch branch (`:743`) and combine branch (`:1063`)
- [ ] (optional) `launch.cpp` cases, kept in sync with the Python names
- [ ] `pip install -e .`, `rm -rf ~/.mori/jit` if in doubt
- [ ] standalone correctness + bench vs. the baseline `KernelType`

SGLang:

- [ ] `_EXPLICIT_KERNEL_TYPES` + `_kernel_type_override()` in `moriep.py`
- [ ] `cfg = _kernel_type_override() or cfg` in `init_mori_op`
- [ ] send/recv-split checks at `:836` / `:915`, only if the kernel is async
- [ ] A/B with `SGLANG_MORI_KERNEL_TYPE` set vs. unset, engine restarted between runs

## Failure-mode cheat sheet

| Symptom | Cause |
|---|---|
| `Kernel function not found in any loaded module: X_bf16` | symbol name ≠ launched name; or `_KERNEL_TYPE_TO_HIP` points at the wrong `.hip` |
| `std::bad_variant_access` | missed a `KernelType::IntraNode` branch in step 4 |
| Loads fine, faults on first peer access | forgot `MORI_DEFINE_GPU_STATES` |
| Edits have no effect | wheel-installed mori — you're editing the repo, it's reading `_jit-sources/` |
| Garbage output only at some topk / EP size | slot/warp tiling assumes `warpSize % topk == 0` |
| `[mori] JIT kernel compilation skipped for 'ep_myexp'` (a *warning*, not an error) | your `.hip` failed to compile — the real hipcc error is in that warning's text |

## Related docs

- `docs/MORI-EP-GUIDE.md` — the operator API, config fields, env vars, tuning
- `docs/MORI-EP-BENCHMARK.md` — benchmark harness
- `docs/MORI_JIT_ARCHITECTURE.md` — JIT compile/cache/load internals
- `docs/PROFILER.md` — the built-in kernel profiler
