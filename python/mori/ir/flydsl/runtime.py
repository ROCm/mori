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
FlyDSL-specific runtime helpers for mori shmem integration.

  - ``get_bitcode_path()``  — returns the path to libmori_shmem_device.bc
  - ``shmem_module_init()`` — initializes ``globalGpuStates`` for a loaded
                              FlyDSL HIP module
"""

from mori.ir.bitcode import find_bitcode

_FLYDSL_COV = 6
_FLYDSL_ROCDL_ABI = 600


def _check_flydsl_rocdl_abi() -> None:
    """Verify FlyDSL still lowers ROCm code objects with ABI 600.

    Mori's FlyDSL device bitcode is compiled with code object version 6.  The
    matching FlyDSL `rocdl-attach-target` option is `abi=600`; if either side
    changes, linked kernels can compile but fail later at HIP module load time.
    """
    try:
        from flydsl.compiler.backends import get_backend

        backend = get_backend()
        fragments = backend.pipeline_fragments(compile_hints={})
    except Exception as exc:
        raise RuntimeError(
            "Unable to inspect FlyDSL ROCm backend ABI settings"
        ) from exc

    attach_fragments = [frag for frag in fragments if "rocdl-attach-target" in frag]
    if not attach_fragments:
        raise RuntimeError("FlyDSL ROCm pipeline has no rocdl-attach-target pass")

    expected = f"abi={_FLYDSL_ROCDL_ABI}"
    if not any(expected in frag for frag in attach_fragments):
        raise RuntimeError(
            "MORI FlyDSL bitcode is built with cov=6 and requires FlyDSL "
            f"rocdl-attach-target {expected}; got: {attach_fragments}"
        )


def get_bitcode_path() -> str:
    """Return the path to libmori_shmem_device.bc (compiled with cov=6 for FlyDSL ABI).

    Usage::

        from mori.ir.flydsl import get_bitcode_path
        bc = get_bitcode_path()
    """
    _check_flydsl_rocdl_abi()
    return find_bitcode(cov=_FLYDSL_COV)


def shmem_module_init(hip_module: int):
    """Initialize globalGpuStates in a FlyDSL-loaded HIP module."""
    import mori.shmem as ms

    return ms.shmem_module_init(hip_module)


def install_hook() -> None:
    """Compatibility no-op.

    Modern FlyDSL integration attaches ``shmem_module_init`` directly through
    ``link_extern(..., module_init_fn=...)`` when constructing the extern
    wrappers, so no global hook installation is required.
    """
    return None


def install_jit_hook() -> None:
    """Compatibility alias for :func:`install_hook`."""
    return install_hook()
