# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""ctypes binding for the JIT v2 plan_api C ABI (``mori_jit_*``, libmori_jit.so).

The Python side of the C++ JIT v2 framework (``include/mori/jit`` + ``src/jit``).
It is the counterpart of ``src/jit/plan_api.cpp``, and it is **op-agnostic**:

  * ``load_library(name)`` dlopens an op-library (e.g. ``libmori_ops_v2.so``) so
    its kernels register into the one shared registry that lives in libmori_jit.
  * ``registered_plans()`` lists what has registered.
  * ``make_plan(kernel)`` builds a Plan class for any registered kernel, generated
    entirely from the two schemas C++ publishes -- the REQUEST schema (constructor
    params, defaults, types) and the ARGS schema (launch params + struct layout,
    whose size is asserted against C++'s own sizeof). There is nothing per-kernel
    in this file: ``make_plan("ep_dispatch")`` works the moment that kernel
    registers, with no edit here.

An EP-specific shim (``mori/ops/dispatch_combine_v2/ep_plans.py``) is what knows
to load libmori_ops_v2.so; this module knows only how to talk to the ABI.

Not pybind11 and not the CPython C API: the boundary carries only pointers and
scalars, so there is nothing for a type-conversion layer to do, and ctypes costs
no build step and no compiled artifact. mori already binds libamdhip64 this way
(``mori.jit.hip_driver``). See docs/MORI_JIT_V2_DESIGN.md §6.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

__all__ = [
    "load_library",
    "make_plan",
    "precompile",
    "registered_plans",
    "library_path",
    "DTYPES",
]

# Must match mori::ops::v2::EpDType; an enum-tagged request field accepts a name
# from here as well as an int. "byte8" is the transport type fp8 and fp4 both map
# to -- dispatch only copies, so one byte covers fp8 directly and fp4 as 2 e2m1
# (the caller halves hiddenDim). Combine cannot use it; C++ rejects that.
DTYPES = {"bf16": 0, "fp32": 1, "byte8": 2}

# The C ABI + the plan registry both live here; op-libraries register INTO it.
_ABI_NAME = "libmori_jit.so"

# Schema type tags -> ctypes. Kept small on purpose: the boundary is pointers
# and scalars, and anything richer belongs in the Cfg, not the arguments.
_CTYPE = {
    "p": ctypes.c_void_p,
    "u64": ctypes.c_uint64,
    "i64": ctypes.c_int64,
    "i32": ctypes.c_int32,
    "f32": ctypes.c_float,
}

_abi: ctypes.CDLL | None = None  # bound handle exposing the mori_jit_* symbols


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------


def _candidate_dirs(extra_dirs=None) -> list[Path]:
    here = Path(__file__).resolve()  # <repo>/python/mori/jit/v2/plan_api.py
    pkg = here.parents[2]  # <...>/mori/
    repo = here.parents[4]  # <repo>/
    dirs = [Path(d) for d in (extra_dirs or [])]
    dirs += [pkg, pkg / "lib"]
    for sub in ("src/jit/v2", "src/ops/dispatch_combine_v2", "lib"):
        dirs.append(repo / "build" / sub)
    if repo.is_dir():
        for sub in ("src/jit/v2", "src/ops/dispatch_combine_v2"):
            for d in sorted(repo.glob(f"build*/{sub}"), reverse=True):
                dirs.append(d)
    # In a build tree the ABI (libmori_jit.so, src/jit/v2) and an op library
    # (src/ops/*) are siblings, so for every candidate also probe its ../../jit/v2.
    dirs += [d.parent.parent / "jit" / "v2" for d in list(dirs)]
    return dirs


def _find(name: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        so = d / name
        if so.is_file():
            return so
    return None


def _ensure_device_nic() -> None:
    """Resolve the NIC identity through v1's single authority before the C++
    toolchain reads it.

    The NIC is a cache-key component (and, for a GDA build, selects which cco
    bitcode links). The C++ side does not probe -- it just reads MORI_DEVICE_NIC --
    so if nothing set it, it would fall back to a bare default. Fill it here from
    mori.jit.config.detect_nic_type() (env -> /sys/class/infiniband -> lspci -> lib
    -> mlx5), the same detector v1 uses, so v2 picks the real NIC when present and
    mlx5 otherwise. Best-effort: if it cannot run, the C++ default (mlx5) applies.
    """
    if os.environ.get("MORI_DEVICE_NIC"):
        return
    try:
        from mori.jit.config import detect_nic_type

        os.environ["MORI_DEVICE_NIC"] = detect_nic_type()
    except Exception:
        pass


def load_library(name: str = _ABI_NAME, extra_dirs=None) -> None:
    """dlopen an op-library so its kernels register into the JIT registry.

    Brings ``libmori_jit.so`` (the ABI + registry) as its dependency; both are
    loaded RTLD_GLOBAL so the registry is one shared instance and the ABI symbols
    resolve regardless of which directory each .so lives in. Idempotent enough to
    call once per op-library at import time. ``extra_dirs`` is searched first --
    a caller pointing at a build tree passes its lib dir there.
    """
    global _abi
    _ensure_device_nic()  # set MORI_DEVICE_NIC before the C++ toolchain resolves
    dirs = _candidate_dirs(extra_dirs)

    # 1) The ABI/registry home. Load it explicitly when we can find it, so ABI
    #    calls have a handle even if the op-library is built in a sibling dir.
    if _abi is None:
        abi = _find(_ABI_NAME, dirs)
        if abi is not None:
            _abi = ctypes.CDLL(str(abi), mode=ctypes.RTLD_GLOBAL)
            _bind(_abi)

    if name == _ABI_NAME:
        if _abi is None:
            raise OSError(
                f"{_ABI_NAME} not found. Build the jit target, or pass its dir. "
                "Searched:\n  " + "\n  ".join(str(d / _ABI_NAME) for d in dirs)
            )
        return

    # 2) The op-library. Loading it registers its kernels; if the ABI was not
    #    found on its own, it comes in here as a dependency (dlsym walks it).
    so = _find(name, dirs)
    if so is None:
        raise OSError(
            f"{name} not found. Build it, or set its lib dir. Searched:\n  "
            + "\n  ".join(str(d / name) for d in dirs)
        )
    ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
    if _abi is None:
        _abi = ctypes.CDLL(str(so))
        _bind(_abi)


def _bind(lib: ctypes.CDLL) -> None:
    """The whole ABI: ten symbols, the same for every kernel."""
    lib.mori_jit_last_error.restype = ctypes.c_char_p
    lib.mori_jit_last_error.argtypes = []

    lib.mori_jit_registered_plans.restype = ctypes.c_char_p
    lib.mori_jit_registered_plans.argtypes = []

    lib.mori_jit_plan_args_schema.restype = ctypes.c_char_p
    lib.mori_jit_plan_args_schema.argtypes = [ctypes.c_char_p]

    lib.mori_jit_plan_request_schema.restype = ctypes.c_char_p
    lib.mori_jit_plan_request_schema.argtypes = [ctypes.c_char_p]

    lib.mori_jit_plan_args_size.restype = ctypes.c_int
    lib.mori_jit_plan_args_size.argtypes = [ctypes.c_char_p]

    lib.mori_jit_plan_create.restype = ctypes.c_void_p
    lib.mori_jit_plan_create.argtypes = [
        ctypes.c_char_p,  # kernel
        ctypes.c_char_p,  # arch
        ctypes.POINTER(ctypes.c_char_p),  # names
        ctypes.POINTER(ctypes.c_longlong),  # values
        ctypes.c_int,  # count
    ]

    lib.mori_jit_plan_destroy.restype = None
    lib.mori_jit_plan_destroy.argtypes = [ctypes.c_void_p]

    lib.mori_jit_plan_launch.restype = ctypes.c_int
    lib.mori_jit_plan_launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    lib.mori_jit_plan_info.restype = ctypes.c_int
    lib.mori_jit_plan_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]

    lib.mori_jit_precompile.restype = ctypes.c_int
    lib.mori_jit_precompile.argtypes = [ctypes.c_char_p, ctypes.c_char_p]


def _load() -> ctypes.CDLL:
    """The bound ABI handle. Auto-loads libmori_jit alone (empty registry) if no
    op-library has been loaded yet, so registered_plans() degrades gracefully."""
    if _abi is None:
        load_library(_ABI_NAME)
    return _abi


def library_path() -> str:
    """Path of the ABI library, for diagnostics."""
    for d in _candidate_dirs():
        if (d / _ABI_NAME).is_file():
            return str(d / _ABI_NAME)
    return _ABI_NAME


def _error() -> str:
    msg = _load().mori_jit_last_error()
    return msg.decode(errors="replace") if msg else "unknown error"


def registered_plans() -> list[str]:
    raw = _load().mori_jit_registered_plans()
    return [s for s in (raw or b"").decode().split(",") if s]


def _as_ptr(x) -> int:
    """A torch tensor, anything with .data_ptr(), an int, or None.

    torch is never imported here: taking .data_ptr() by duck typing is what keeps
    this layer -- and every C++ header behind it -- free of a torch dependency.
    """
    if x is None:
        return 0
    if hasattr(x, "data_ptr"):
        return int(x.data_ptr())
    if isinstance(x, int):
        return x
    raise TypeError(
        f"expected a tensor, an int pointer, or None; got {type(x).__name__}"
    )


# ---------------------------------------------------------------------------
# The factory. Nothing below knows about any particular kernel: the REQUEST schema
# gives the constructor its parameters, the ARGS schema gives launch its struct
# layout, and both come from C++ -- so registering a kernel there is the whole job.
#
# The only two conventions:
#   * a request field tagged `e` (enum) accepts a name from DTYPES as well as an int
#   * request fields under `off.` are region offsets: pass `arena=` and they are
#     bound once from arena.offset(region)
# ---------------------------------------------------------------------------


def _camel_to_snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _request_schema(kernel: str) -> dict:
    lib = _load()
    raw = lib.mori_jit_plan_request_schema(kernel.encode())
    if not raw:
        raise RuntimeError(f"mori jit: {_error()}")
    schema = {}
    for item in raw.decode().split(","):
        if not item:
            continue
        head, _, default = item.partition("=")
        name, _, tag = head.partition(":")
        schema[name] = (tag, int(default))
    return schema


def _args_struct(kernel: str):
    """Build the argument struct from the schema C++ publishes, and check it."""
    lib = _load()
    raw = lib.mori_jit_plan_args_schema(kernel.encode())
    if not raw:
        raise RuntimeError(f"mori jit: {_error()}")
    fields = []
    for item in raw.decode().split(","):
        if not item:
            continue
        name, _, tag = item.partition(":")
        if tag not in _CTYPE:
            raise RuntimeError(
                f"mori jit: kernel '{kernel}' schema has unknown type '{tag}'"
            )
        fields.append((name, _CTYPE[tag]))

    struct = type(f"_{kernel}Args", (ctypes.Structure,), {"_fields_": fields})
    want = lib.mori_jit_plan_args_size(kernel.encode())
    if ctypes.sizeof(struct) != want:
        raise RuntimeError(
            f"mori jit: kernel '{kernel}' args are {ctypes.sizeof(struct)}B from the schema "
            f"but {want}B in C++ -- the schema string and the struct disagree"
        )
    return struct, [f[0] for f in fields]


def _coerce(tag: str, name: str, value) -> int:
    if tag == "e":
        return _enum_code(value)
    if tag == "b":
        return 1 if value else 0
    return int(value)


def _enum_code(value) -> int:
    """An enum field accepts an int, a DTYPES name, or a torch dtype."""
    if isinstance(value, int):
        return value
    name = (getattr(value, "name", None) or str(value)).rsplit(".", 1)[-1].lower()
    name = {
        "bfloat16": "bf16",
        "float32": "fp32",
        "float": "fp32",
        # every sub-16-bit dtype is transported as raw bytes; see DTYPES
        "float8_e4m3fn": "byte8",
        "float8_e4m3fnuz": "byte8",
        "float4_e2m1fn_x2": "byte8",
    }.get(name, name)
    if name not in DTYPES:
        raise ValueError(
            f"unsupported dtype {value!r}; expected one of {sorted(DTYPES)}"
        )
    return DTYPES[name]


def make_plan(kernel: str) -> type:
    """A Plan class for a kernel registered on the C++ side, generated from its
    published schemas. Nothing about the kernel is written here."""

    req_schema = _request_schema(kernel)
    args_t, arg_names = _args_struct(kernel)

    # Caller-facing names: snake_case, and the arena-offset group folded into
    # `arena=`. A launch argument named `off<Region>` is an arena region offset;
    # passing `arena=` binds it once instead of repeating it every call.
    snake_to_wire = {_camel_to_snake(n): n for n in req_schema}
    arg_regions = {
        n: n[3].lower() + n[4:]
        for n in arg_names
        if len(n) > 3 and n.startswith("off") and n[3].isupper()
    }
    arg_snake_to_wire = {_camel_to_snake(n): n for n in arg_names}
    _all_regions = sorted(arg_regions.values())

    class Plan:
        _kernel = kernel
        _args_t = args_t
        _arg_names = arg_names

        def __init__(self, *, arch=None, arena=None, region_names=None, **kwargs):
            lib = _load()
            req = {}
            for snake, wire in snake_to_wire.items():
                if snake in kwargs:
                    req[wire] = _coerce(req_schema[wire][0], wire, kwargs.pop(snake))
                elif wire in kwargs:  # the C++ spelling also works
                    req[wire] = _coerce(req_schema[wire][0], wire, kwargs.pop(wire))

            if kwargs:
                raise TypeError(
                    f"{kernel}: unknown argument(s) {sorted(kwargs)}; "
                    f"accepted: {sorted(snake_to_wire)} (+ arch, arena, region_names)"
                )

            items = list(req.items())
            n = len(items)
            cnames = (ctypes.c_char_p * n)(*[k.encode() for k, _ in items])
            cvalues = (ctypes.c_longlong * n)(*[int(v) for _, v in items])
            handle = lib.mori_jit_plan_create(
                kernel.encode(), arch.encode() if arch else None, cnames, cvalues, n
            )
            if not handle:
                raise RuntimeError(f"mori jit [{kernel}]: {_error()}")
            self._handle = ctypes.c_void_p(handle)
            self._arena = arena  # held: the kernel dereferences the window
            # Known at construction, so the caller need not repeat them per launch.
            self._defaults = {}
            if "window" in arg_names:
                self._defaults["window"] = int(arena.handle) if arena is not None else 0
            # Runtime region offsets: bound once here, not repeated per launch.
            if arena is not None:
                names = region_names or {}
                for wire, region in arg_regions.items():
                    self._defaults[wire] = int(arena.offset(names.get(region, region)))

        def bind(self, **args) -> None:
            """Pin launch arguments that never change between calls.

            The arena window and its region offsets are bound this way at
            construction; `rank` and anything else fixed for the op's lifetime
            belongs here too, so `launch()` only carries what actually varies.
            """
            for k, v in args.items():
                wire = arg_snake_to_wire.get(k, k)
                if wire not in arg_names:
                    raise TypeError(
                        f"{kernel}: unknown launch argument '{k}'; schema is {arg_names}"
                    )
                self._defaults[wire] = v

        def launch(self, stream=0, **args) -> None:
            """Arguments by name, snake_case or the C++ spelling, per the schema."""
            if self._handle is None:
                raise RuntimeError("launch on a closed plan")
            merged = dict(self._defaults)
            for k, v in args.items():
                wire = arg_snake_to_wire.get(k, k)
                if wire not in arg_names:
                    raise TypeError(
                        f"{kernel}: unknown launch argument '{k}'; schema is {arg_names}"
                    )
                merged[wire] = v
            buf = args_t()
            for name in arg_names:
                if name in merged:
                    setattr(buf, name, _as_ptr(merged[name]))
            rc = _load().mori_jit_plan_launch(
                self._handle,
                ctypes.byref(buf),
                ctypes.sizeof(buf),
                ctypes.c_void_p(_as_ptr(stream)),
            )
            if rc != 0:
                raise RuntimeError(f"mori jit [{kernel}] launch: {_error()}")

        __call__ = launch

        @property
        def info(self) -> dict:
            """What C++ resolved: geometry, every Cfg field, and the cache dir."""
            lib = _load()
            need = lib.mori_jit_plan_info(self._handle, None, 0)
            if need < 0:
                raise RuntimeError(f"mori jit [{kernel}] info: {_error()}")
            buf = ctypes.create_string_buffer(need)
            lib.mori_jit_plan_info(self._handle, buf, need)
            out = {}
            for line in buf.value.decode().splitlines():
                k, _, v = line.partition("=")
                if not k:
                    continue
                if v in ("true", "false"):
                    out[k] = v == "true"
                elif v.lstrip("-").isdigit():
                    out[k] = int(v)
                else:
                    out[k] = v
            return out

        def close(self) -> None:
            if getattr(self, "_handle", None) is not None:
                _load().mori_jit_plan_destroy(self._handle)
                self._handle = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

        def __del__(self):
            try:
                self.close()
            except Exception:
                pass

        def __repr__(self) -> str:
            i = self.info
            return (
                f"{type(self).__name__}(grid={i.get('gridX')}, block={i.get('blockX')}, "
                f"lds={i.get('sharedBytes')}B)"
            )

    Plan.__name__ = "".join(w.capitalize() for w in kernel.split("_")) + "Plan"
    Plan.__qualname__ = Plan.__name__
    Plan.__doc__ = (
        f"JIT plan for the '{kernel}' kernel, generated from its C++ schema.\n\n"
        f"Constructor: {', '.join(sorted(snake_to_wire))}"
        + (f", arena=, region_names= (regions: {_all_regions})" if _all_regions else "")
        + f"\nLaunch: {', '.join(_camel_to_snake(a) for a in arg_names)}"
    )
    return Plan


def precompile(kernel: str, arch: str | None = None) -> int:
    """Fill the disk cache for `arch` (None = the local device). No GPU touched."""
    n = _load().mori_jit_precompile(kernel.encode(), arch.encode() if arch else None)
    if n < 0:
        raise RuntimeError(f"mori jit precompile [{kernel}]: {_error()}")
    return n
