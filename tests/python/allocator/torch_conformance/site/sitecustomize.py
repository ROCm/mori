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
"""Force a torch symm_mem backend in every interpreter, including spawned test workers.

Active only when MORI_CONFORMANCE=1; run_conformance.py puts this directory on PYTHONPATH.

MORI_CONFORMANCE_BACKEND  backend to select (default MORI; CUDA gives a control run)
MORI_CONFORMANCE_VIEW     which of torch's ROCm skips to lift (default C):
                            A  none, as shipped
                            B  ROCm < 10.1 version gates and MI350 (gfx950) arch skips
                            C  B plus every skip_if_rocm_multiprocess
MORI_CONFORMANCE_LIFTED   file a test appends to when it runs only because view C lifted
                          its skip_if_rocm_multiprocess (lets the runner derive view B)
MORI_CONFORMANCE_PG_TIMEOUT  seconds MultiProcContinuousTest waits on a stuck peer (40)
MORI_CONFORMANCE_AS_CUDA  1: get_backend() answers "CUDA" for the forced backend, for tests
                          that gate a generic contract on that name (stream serialization)
MORI_CONFORMANCE_PREFIX   sys.prefix of the environment under test; other interpreters
                          (a tool's python3 on PATH) are left alone

Processes started while this runs (the backend's JIT probes the GPU with a Python tool)
skip it: they would import the backend again, and so recurse.
"""

import os
import sys


def _lift_skips(view):
    import functools

    import torch.testing._internal.common_distributed as cd
    import torch.testing._internal.common_utils as cu

    def keep(*_args, **_kwargs):
        return lambda f: f

    cd.skip_if_rocm_ver_lessthan_multiprocess = keep
    cd.skip_if_rocm_arch_multiprocess = keep
    if hasattr(cu, "lazy_skip_if"):
        orig_lazy_skip_if = cu.lazy_skip_if

        def lazy_skip_if(cond, reason):
            if "MI350" in reason or "ROCm 10.1" in reason:
                return lambda f: f
            return orig_lazy_skip_if(cond, reason)

        cu.lazy_skip_if = lazy_skip_if
    if view != "C":
        return
    lifted = os.environ.get("MORI_CONFORMANCE_LIFTED")

    def skip_if_rocm_multiprocess(func):
        if not lifted:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with open(lifted, "a") as f:
                f.write(f"{func.__qualname__}\n")
            return func(*args, **kwargs)

        return wrapper

    cd.skip_if_rocm_multiprocess = skip_if_rocm_multiprocess


def _report_worker_exceptions():
    # A worker's exception is lost when its peers hang; print it where the log keeps it.
    import traceback
    import unittest
    from datetime import timedelta

    import torch.testing._internal.common_distributed as cd

    mpct = cd.MultiProcContinuousTest
    orig = mpct._run_test_given_id.__func__

    def run(cls, test_id, **kwargs):
        try:
            return orig(cls, test_id, **kwargs)
        except unittest.SkipTest:
            raise
        except BaseException:
            rank = getattr(cls, "rank", "?")
            print(
                f"[conformance worker rank={rank}] EXCEPTION in {test_id}:\n"
                + traceback.format_exc(),
                file=sys.stderr,
                flush=True,
            )
            raise

    mpct._run_test_given_id = classmethod(run)
    mpct.timeout = timedelta(
        seconds=int(os.environ.get("MORI_CONFORMANCE_PG_TIMEOUT", "40"))
    )


def _select_backend():
    import torch.distributed._symmetric_memory as symm_mem

    backend = os.environ.get("MORI_CONFORMANCE_BACKEND", "MORI")
    if backend == "MORI":
        import mori.allocator

        mori.allocator.register_symm_backend()
    symm_mem.set_backend(backend)
    if os.environ.get("MORI_CONFORMANCE_AS_CUDA") == "1":
        get_backend = symm_mem.get_backend

        def as_cuda(device):
            name = get_backend(device)
            return "CUDA" if name == backend else name

        symm_mem.get_backend = as_cuda


_GUARD = "MORI_CONFORMANCE_STARTING"


def _wanted():
    if os.environ.get("MORI_CONFORMANCE") != "1" or os.environ.get(_GUARD):
        return False
    prefix = os.environ.get("MORI_CONFORMANCE_PREFIX")
    return not prefix or os.path.realpath(sys.prefix) == os.path.realpath(prefix)


if _wanted():
    os.environ[_GUARD] = "1"
    try:
        view = os.environ.get("MORI_CONFORMANCE_VIEW", "C")
        if view not in ("A", "B", "C"):
            raise ValueError(f"MORI_CONFORMANCE_VIEW={view!r}, expected A, B or C")
        if view != "A":
            _lift_skips(view)
        _report_worker_exceptions()
        _select_backend()
    except Exception as e:  # loud: a silent fallback would test another backend
        print(
            f"[conformance pid={os.getpid()}] cannot select the backend: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        raise
    finally:
        del os.environ[_GUARD]
