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
"""EP kernels' JIT plans.

The only EP-specific part of the C++/JIT binding: load ``libmori_ops_v2.so`` so
its kernels register into the shared JIT registry, then expose the Plan classes:
the intranode ``EpDispatchPlan`` / ``EpCombinePlan``, plus the eight internode
passes in ``EP_INTERNODE_PLANS``. Everything else -- the ABI, the Plan factory,
schema-driven arg structs -- is generic and lives in ``mori.jit.v2.plan_api``.

Importing this module loads the library (raising OSError if it is not built),
which is what lets the binding test skip cleanly when the .so is absent.
``MORI_V2_LIB_DIR`` points at a build tree; the package dir is the fallback.
"""

from __future__ import annotations

import os
import sys
import warnings
import weakref

from mori.jit.v2 import plan_api

# EP naming/library live here; the generic layer stays op-agnostic.
_LIB_NAME = "libmori_ops_v2.so"
DTYPES = plan_api.DTYPES
make_plan = plan_api.make_plan
registered_plans = plan_api.registered_plans
precompile = plan_api.precompile
library_path = plan_api.library_path


def _extra_dirs() -> list[str]:
    env = os.environ.get("MORI_V2_LIB_DIR")
    return [env] if env else []


plan_api.load_library(_LIB_NAME, extra_dirs=_extra_dirs())

_EpDispatchJitPlan = make_plan("ep_dispatch")

# Staging for unbound dispatch plans, per device and layout; held by its plans.
_FALLBACK_STAGING = weakref.WeakValueDictionary()

# A warning about a caller's plan points at the first frame outside these.
_BINDING_DIRS = tuple(
    os.path.join(os.path.dirname(f), "") for f in (plan_api.__file__, __file__)
)


class EpDispatchPlan(_EpDispatchJitPlan):
    # gfx125x: staging_base defaults to a buffer shared per layout until bound (#703).
    __doc__ = _EpDispatchJitPlan.__doc__

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fallback_staging = None
        info = self.info
        # Only the gfx125x body stages; EpDispatchSpec::LaunchRaw tests the same.
        if info["sharedBytes"]:
            self.bind(staging_base=_FallbackStaging(self, info))

    def close(self) -> None:
        super().close()
        self._fallback_staging = None


class _FallbackStaging:
    """The staging_base an EpDispatchPlan starts with; resolved at its first launch."""

    __slots__ = ("_plan", "_info")

    def __init__(self, plan, info):
        self._plan = weakref.ref(plan)
        self._info = info

    def data_ptr(self) -> int:
        buf = _fallback_staging(self._info)
        plan = self._plan()
        if plan is not None:
            plan._fallback_staging = buf
            # An int from here on: the launch struct caches it like a caller's.
            plan.bind(staging_base=buf.data_ptr())
        return buf.data_ptr()

    def group_key(self):
        # What a launch group compares, without allocating: the buffer's address
        # once its layout has one, the layout until then.
        key = _staging_key(self._info)
        buf = _FALLBACK_STAGING.get(key)
        return buf.data_ptr() if buf is not None else ("fallback", *key)


def _staging_key(info: dict) -> tuple:
    import torch

    # The cap EpMaxRecv resolves, so maxRecv=0 and its resolved value share.
    max_recv = info["maxRecv"]
    recv = max_recv if max_recv > 0 else info["worldSize"] * info["maxTokPerRank"]
    return (
        torch.cuda.current_device(),
        info["worldSize"],
        recv,
        info["scaleBytes"],
        info["maxTokPerRank"],
        bool(info["aggReserve"]),
    )


def _fallback_staging(info: dict):
    import torch

    from mori.ops.dispatch_combine_v2.hip_backend import ep_staging_bytes

    key = _staging_key(info)
    device = key[0]
    buf = _FALLBACK_STAGING.get(key)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "ep_dispatch: staging_base is not bound, and the buffer it falls back "
                "to cannot be allocated inside a graph capture; bind one "
                "(ep_staging_bytes) or launch once before capturing"
            )
        nbytes = ep_staging_bytes(*key[1:])
        buf = _FALLBACK_STAGING[key] = torch.zeros(
            nbytes, dtype=torch.uint8, device=device
        )
        warnings.warn(
            f"ep_dispatch: staging_base is not bound; allocated {nbytes:,} bytes for "
            "it. Allocate ep_staging_bytes() bytes once, keep them alive, and "
            "bind(staging_base=buf.data_ptr()) on every dispatch plan "
            "(ROCm/mori#703); this fallback will be removed.",
            FutureWarning,
            stacklevel=_caller_stacklevel(),
        )
    return buf


def _caller_stacklevel() -> int:
    # The caller's warnings.warn stacklevel for the first frame outside mori.
    level, frame = 1, sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_BINDING_DIRS):
        level, frame = level + 1, frame.f_back
    return level


EpCombinePlan = make_plan("ep_combine")


# The internode sequence. Eight plans rather than two: its dispatch and combine
# are several passes each, and each pass is its own module. Both name tables must
# match the C++ enums, which are now v2's own -- EpInterNodeDType and
# EpQuantType, both in ep_internode_cfg.hpp.
INTERNODE_DTYPES = {"bf16": 0, "f32": 1, "fp8_fnuz": 2, "fp8_ocp": 3, "fp4": 4}
INTERNODE_QUANT_TYPES = {
    "none": 0,
    "fp8directcast": 1,
    "fp8blockwisequant": 2,
    "fp4blockwisequant": 3,
}
_INTERNODE_ENUMS = {"dtype": INTERNODE_DTYPES, "quantType": INTERNODE_QUANT_TYPES}

# Keyed by pass name: a caller drives them as a sequence.
EP_INTERNODE_PLANS = {
    name: make_plan(f"ep_internode_{name}", enums=_INTERNODE_ENUMS)
    for name in (
        "copystaging",
        "dispatch",
        "dispatch_ll",
        "combinesync",
        "combinesyncbarrier",
        "combine",
        "combine_ll",
        "combineall",
    )
}

# AsyncLL. Same Request and Args schema as the internode passes -- it is
# specialised on the same shape and launched against the same struct -- so the
# same enum map applies. Nine passes: five for dispatch (three of them the send
# half, two the recv half) and four for combine.
EP_ASYNCLL_PLANS = {
    name: make_plan(f"ep_asyncll_{name}", enums=_INTERNODE_ENUMS)
    for name in (
        "dispatch_slotassign",
        "dispatch_sendcopy",
        "dispatch_sendtransfer",
        "dispatch_recvtransfer",
        "dispatch_recvcopy",
        "combine_sendcopy",
        "combine_sendtransfer",
        "combine_recvtransfer",
        "combine_recvcopy",
        # The LSA transport's transfer passes. Same protocol, peer stores instead
        # of the copy engine; the op picks one set or the other at build time.
        # dispatch_sendcopy_lsa replaces the sendcopy/sendtransfer PAIR: with no
        # engine to feed there is no reason to stage, so the gather lands in the
        # peer and one pass over the payload disappears.
        "dispatch_sendcopy_lsa",
        "dispatch_sendtransfer_lsa",
        "dispatch_recvtransfer_lsa",
        "combine_sendtransfer_lsa",
        "combine_recvtransfer_lsa",
    )
}
