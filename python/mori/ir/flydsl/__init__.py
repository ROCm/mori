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
"""
mori.ir.flydsl — FlyDSL integration for mori shmem device API.

Recommended usage (qualified access keeps the lazy-init guarantee)::

    from mori.ir import flydsl as mori_shmem

    @flyc.kernel
    def my_kernel(buf: fx.Tensor):
        pe = mori_shmem.my_pe()
        mori_shmem.putmem_nbi_warp(buf, buf, 64, (pe + 1) % 2, 0)
        mori_shmem.quiet_thread_pe((pe + 1) % 2)

.. note::

   Avoid ``from mori.ir.flydsl import *`` — it iterates ``__all__`` and
   eagerly materialises every :class:`ExternFunction`, which defeats
   the lazy-init design and forces the ~20s cold-cache bitcode build
   (see :func:`prepare`) on import.  Use the qualified form above or
   explicit names (``from mori.ir.flydsl import my_pe, n_pes``).

Architecture:

* :mod:`mori.ir.flydsl.compile_helper` is the single source of truth
  for mori-side compile metadata (bitcode path, COV, module-init
  callable).
* :mod:`mori.ir.flydsl.ops` constructs one :class:`ExternFunction`
  per shmem device function, baking the metadata in at build time.
* FlyDSL's compiler picks the metadata up through
  ``CompilationContext`` populated by :class:`ExternFunction`, so no
  ``mori`` import ever happens on FlyDSL's JIT path.
"""

from mori.ir.ops import SIGNAL_SET, SIGNAL_ADD  # cheap constants, no FlyDSL trigger

from . import ops as _ops  # importing the submodule does NOT call _ensure_ops


def prepare() -> None:
    """Eagerly resolve mori shmem bitcode and build every ExternFunction.

    Call this once at process startup (e.g. in ``main()``) so the first
    kernel launch does not pay the cold-cache cost of JIT-compiling
    ``libmori_shmem_device.bc``.

    Cost model
    ----------
    * Cold JIT cache: ~15-20s (hipcc compiles the **mori shmem device
      bitcode**, not FlyDSL's kernel — see ``mori.jit.core.ensure_bitcode``).
    * Warm cache (same arch / NIC / cov on the machine): microseconds.
    * The cache is shared across processes and users that share the
      same ``MORI_CACHE_DIR`` (default ``~/.cache/mori/...``).

    Operationally you may prefer to warm the cache offline via
    ``MORI_PRECOMPILE=1 python -c 'import mori'`` during image build /
    CI; :func:`prepare` is the in-process equivalent and is safe to
    call multiple times (idempotent — the underlying caches make
    subsequent calls a no-op).
    """
    _ops._ensure_ops()


def __getattr__(name: str):
    """Forward attribute access to the ops submodule.

    Keeping this layer — instead of ``from .ops import *`` at import
    time — is what makes the whole package truly lazy: merely running
    ``import mori.ir.flydsl`` never materialises any
    :class:`ExternFunction` (and therefore never forces the cold-cache
    bitcode JIT).  Only the first explicit access to an op name
    triggers :func:`ops._ensure_ops`.

    Once resolved we cache the value in the package globals so
    subsequent accesses skip this hook entirely.
    """
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    val = getattr(_ops, name)
    globals()[name] = val
    return val


def __dir__():
    return sorted(set(list(globals()) + list(_ops.__all__)))


__all__ = list(_ops.__all__) + ["prepare"]
