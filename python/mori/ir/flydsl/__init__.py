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
   the lazy-init design and forces the one-time mori shmem bitcode
   build (see :func:`prepare`) on import.  Use the qualified form
   above or explicit names (``from mori.ir.flydsl import my_pe, n_pes``).

Architecture:

* :mod:`mori.ir.flydsl.compile_helper` is the single source of truth
  for mori-side compile metadata (bitcode path, COV, module-init
  callable).
* :mod:`mori.ir.flydsl.ops` constructs one :class:`ExternFunction`
  per shmem device function, baking the metadata in at build time.
* FlyDSL's compiler picks the metadata up through
  ``CompilationContext`` populated by :class:`ExternFunction`, so no
  ``mori`` import ever happens on FlyDSL's JIT path.

Pickling / disk cache contract
------------------------------

:class:`ExternFunction` instances (i.e. ``mori.ir.flydsl.my_pe``,
``n_pes``, ``putmem_nbi_warp``, ...) are **not picklable**.  They are
module-level singletons built from ``MORI_DEVICE_FUNCTIONS`` and are
meant to be looked up fresh in each process via ``import`` / attribute
access.

FlyDSL's ``CompiledArtifact`` disk cache does **not** rely on pickling
these objects.  It only serialises the ``module_init_fn`` callable
(here, ``mori.shmem.shmem_module_init``) by its fully-qualified
``module:qualname`` string, and re-imports it on cache hit.  As long as
mori's module-init callable stays a top-level function (not a
``functools.partial`` / lambda / bound method), the disk cache round-
trips correctly.  See ``FlyDSL/python/flydsl/compiler/jit_executor.py``
for the callable-serialisation helpers.
"""

from mori.ir.ops import SIGNAL_SET, SIGNAL_ADD  # cheap constants, no FlyDSL trigger

from . import ops as _ops  # importing the submodule does NOT call _ensure_ops


def prepare() -> None:
    """Eagerly resolve mori shmem bitcode and build every ExternFunction.

    Call this once at process startup (e.g. in ``main()``) so the
    first kernel launch does not pay the one-time cost of building
    ``libmori_shmem_device.bc``.

    What :func:`prepare` actually does
    ----------------------------------

    1. ``mori.jit.core.ensure_bitcode`` — invokes ``hipcc`` to compile
       the mori shmem device C++ sources into ``libmori_shmem_device.bc``
       once per (arch, NIC, COV).  The resulting artifact is a
       **mori-side** build product shared across *every* FlyDSL kernel
       that subsequently calls a mori shmem op.
    2. Instantiates one :class:`ExternFunction` per entry in
       ``MORI_DEVICE_FUNCTIONS`` so later attribute lookups on this
       package (``mori.ir.flydsl.my_pe`` etc.) are pure dict lookups.

    This is *not* FlyDSL's per-kernel JIT cache.  FlyDSL's MLIR→LLVM
    JIT is orthogonal and paid lazily the first time each
    ``@flyc.kernel`` is invoked; :func:`prepare` only warms the
    **mori-side** bitcode that every such kernel will later link
    against via ``link_libs``.

    Cost model
    ----------

    * Cold path (no ``libmori_shmem_device.bc`` on disk for this
      arch/NIC/COV): ~15-20 s, dominated by ``hipcc``.  Intentionally
      off the kernel-launch critical path so that the first
      ``@flyc.kernel`` call stays fast.
    * Warm path (cache hit): microseconds.  :func:`prepare` is
      idempotent — repeated calls are dict lookups after the first.
    * The bitcode cache lives under ``MORI_CACHE_DIR`` (default
      ``~/.cache/mori/...``) and is shared across processes and users
      on the same host.

    Alternatively, warm the cache offline during image build / CI
    with ``MORI_PRECOMPILE=1 python -c 'import mori'``.
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
