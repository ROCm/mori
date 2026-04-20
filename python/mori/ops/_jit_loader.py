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
"""Shared JIT compilation and HIP module loading utilities.

Both ``dispatch_combine`` and ``local_expert_count`` use the same two-step
pattern — compile a ``.hip`` source to ``.hsaco``, then load and initialise it.
The two global dicts here act as the single source of truth so that:

  - Multiple ``EpDispatchCombineOp`` instances for the same kernel type share
    one ``HipModule`` rather than each loading their own.
  - ``local_expert_count`` participates in the same cache instead of
    maintaining its own module-level state.

Pybind imports are deferred via ``sys.modules`` to avoid the circular import
that arises from ``mori.cpp`` ↔ ``mori.ops`` during package init.
"""
import warnings

# hip_name → compiled .hsaco path
_compiled_hsaco: dict[str, str] = {}

# hip_name → loaded-and-initialised HipModule (global singletons)
_loaded_modules: dict = {}


def ensure_compiled(hip_name: str) -> None:
    """Compile *hip_name* to ``.hsaco`` and cache the path in ``_compiled_hsaco``.

    No-op if already compiled.  Emits a warning and leaves the entry absent
    from ``_compiled_hsaco`` on failure so callers can detect it.
    """
    if hip_name in _compiled_hsaco or hip_name in _loaded_modules:
        return
    try:
        from mori.jit.core import compile_genco

        _compiled_hsaco[hip_name] = compile_genco(hip_name)
    except Exception as e:
        warnings.warn(f"[mori] JIT kernel compilation skipped for '{hip_name}': {e}")


def load_hip_module(hip_name: str, init_shmem: bool = True):
    """Load and initialise the HIP module for *hip_name*, cached in ``_loaded_modules``.

    Returns the cached ``HipModule`` on subsequent calls.
    Raises ``RuntimeError`` if the kernel has not been compiled yet
    (i.e. ``ensure_compiled`` must be called first).

    Args:
        init_shmem: If ``True``, call ``shmem_module_init`` after loading.
            Set to ``False`` for kernels that do not use shared memory state
            (e.g. ``local_expert_count``).
    """
    if hip_name in _loaded_modules:
        return _loaded_modules[hip_name]

    hsaco = _compiled_hsaco.get(hip_name)
    if hsaco is None:
        raise RuntimeError(
            f"[mori] Kernel '{hip_name}' not compiled. Call ensure_compiled first."
        )

    import sys

    from mori.jit.hip_driver import HipModule

    mod = HipModule(hsaco)
    if init_shmem:
        sys.modules["libmori_pybinds"].shmem_module_init(mod._module.value)
    _loaded_modules[hip_name] = mod
    return mod
