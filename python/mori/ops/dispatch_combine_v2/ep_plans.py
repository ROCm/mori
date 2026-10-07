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

_EpDispatchPlanBase = make_plan("ep_dispatch")
EpCombinePlan = make_plan("ep_combine")


def self_first_state_bytes(world_size: int) -> int:
    """Size of the state a selfFirst dispatch keeps in its own window.

    The host-side copy of EpSelfFirstBytes in ep_cfg.hpp: one 64 B inbox slot per
    rank rounded up to a 128 B line, then one line for this rank's counter and call
    number and one for its publish ticket.
    """
    inbox = 64 * int(world_size)
    return (inbox + 127) // 128 * 128 + 256


class SelfFirstState:
    """The selfFirst dispatch's symmetric state: one window per arena, owned here.

    Allocated on the communicator that registered the arena's window, so whoever
    builds the arena -- this package's op or a caller driving EpDispatchPlan
    directly -- does not lay it out. Every dispatch plan on one arena shares it:
    ranks may pick different geometries for the same call, and they must still
    meet in one inbox and one call number. Zero when created; reset() of the op
    zeroes it again. Freed when the last plan holding it closes.
    """

    _by_window: dict = {}

    def __init__(self, comm, nbytes: int, key: int):
        self._nbytes = nbytes
        self._key = key
        self._refs = 0
        self._mem = comm.alloc_mem(nbytes)
        self._win = None
        try:
            self._win = comm.register_window(self._mem.ptr, nbytes)
            self.base, self.stride = self._flat_layout()
            self.zero()
        except BaseException:
            if self._win is not None:
                self._win.close()
            self._mem.close()
            raise

    @classmethod
    def acquire(cls, arena, world_size: int) -> "SelfFirstState":
        key = int(arena.handle)
        st = cls._by_window.get(key)
        if st is None:
            from mori.cco import communicator_of_window

            comm = communicator_of_window(key)
            if comm is None:
                raise RuntimeError(
                    "selfFirst dispatch: the arena's window was not registered through "
                    "mori.cco, so there is no communicator to allocate the selfFirst "
                    "state on (or pass self_first=0)"
                )
            st = cls(comm, self_first_state_bytes(world_size), key)
            cls._by_window[key] = st
        st._refs += 1
        return st

    @classmethod
    def of(cls, arena) -> "SelfFirstState | None":
        """The live state of `arena`, without taking a reference."""
        return cls._by_window.get(int(arena.handle))

    def _flat_layout(self) -> tuple[int, int]:
        """(base, stride) such that PE p's copy is at base + p * stride.

        Read once from the window's device descriptor (ccoWindowDevice in cco.hpp:
        u64 winBase, u32 stride4G, i32 lsaRank), so the kernel addresses the state
        without loading it. Checked against the local address, which cco reports
        independently: a descriptor laid out differently fails here, not on the GPU.
        """
        import struct

        import torch

        from mori.tensor_utils import from_gpu_ptr

        raw = bytes(from_gpu_ptr(self._win.handle, (16,), torch.uint8).cpu().tolist())
        win_base, stride4g, lsa_rank = struct.unpack_from("<QIi", raw)
        stride = stride4g << 32
        if stride == 0 or win_base + lsa_rank * stride != self._win.local_ptr:
            raise RuntimeError(
                f"selfFirst state: window descriptor (winBase={win_base:#x}, "
                f"stride4G={stride4g}, lsaRank={lsa_rank}) does not locate the local "
                f"copy at {self._win.local_ptr:#x}"
            )
        return win_base, stride

    def zero(self) -> None:
        import torch

        from mori.tensor_utils import from_gpu_ptr

        if self._win is None:
            raise RuntimeError("SelfFirstState.zero() after its last plan closed")
        from_gpu_ptr(self._win.local_ptr, (self._nbytes,), torch.int8).zero_()
        torch.cuda.synchronize()

    def release(self) -> None:
        self._refs -= 1
        if self._refs > 0:
            return
        type(self)._by_window.pop(self._key, None)
        win, mem, self._win, self._mem = self._win, self._mem, None, None
        win.close()
        mem.close()


def self_first_enabled(slot_word_in_window: bool = True) -> bool:
    """selfFirst wherever the slot allocator word lives in the cco window, the default.

    selfFirst saves the RMW each rank makes on its own slot word, which costs
    microseconds when the word is in the cco window on memory mapped MTYPE_RW. In
    TokOffExt's separate allocation (hip backend, single host, MORI_EP_TOKOFF_EXT=1)
    that local atomic costs nothing and the per-call publish is pure overhead.
    """
    return slot_word_in_window


class EpDispatchPlan(_EpDispatchPlanBase):
    # The generated signature stays the documentation; this only adds the default.
    __doc__ = (_EpDispatchPlanBase.__doc__ or "") + (
        "\n\n``self_first`` defaults to on. When the resolved Cfg"
        "\nhas it (gfx125x dispatch) and an arena is given, the plan binds the arena's"
        "\nSelfFirstState, allocating it on first use; an arena whose window mori.cco"
        "\ncannot trace to a communicator makes construction raise."
    )

    def __init__(self, **kwargs):
        if "self_first" not in kwargs and "selfFirst" not in kwargs:
            kwargs["self_first"] = int(self_first_enabled())
        arena = kwargs.get("arena")
        self._self_first_state = None
        super().__init__(**kwargs)
        if arena is None:
            return
        info = self.info
        if not info.get("selfFirst"):
            return
        try:
            self._self_first_state = SelfFirstState.acquire(arena, info["worldSize"])
        except BaseException:
            super().close()
            raise
        state = self._self_first_state
        self.bind(sf_base=state.base, sf_stride=state.stride)

    def close(self) -> None:
        state, self._self_first_state = getattr(self, "_self_first_state", None), None
        try:
            super().close()
        finally:
            if state is not None:
                state.release()


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
