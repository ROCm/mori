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


def agg_reserve_state_bytes(world_size: int) -> int:
    """Size of the state the blocks of an aggReserve dispatch meet in.

    The host-side copy of EpResStateBytes in ep_cfg.hpp: a 128 B line for the call
    number, a 4 B count per rank rounded up to a line, then a 64 B published word
    per rank.
    """
    w = int(world_size)
    return 128 + (4 * w + 127) // 128 * 128 + 64 * w


_STAGING_ALIGN = 128


def _staging_buffers(world_size: int) -> tuple[tuple[str, int], ...]:
    """What mori keeps in SYMMETRIC memory beyond the arena a caller lays out.

    Empty: the aggReserve state is local memory (EpArgs::resState), which mori
    allocates per plan and a caller never has to provide. A buffer that has to be
    symmetric -- one peers address, or one every plan on an arena must share --
    is added here, and both the bytes the caller reserves and the offsets mori
    cuts the region at follow from it.
    """
    return ()


def staging_layout(world_size: int) -> tuple[dict[str, int], int]:
    """({buffer: offset within mori's staging region}, region bytes)."""
    offsets, off = {}, 0
    for name, nbytes in _staging_buffers(world_size):
        offsets[name] = off
        off += (nbytes + _STAGING_ALIGN - 1) // _STAGING_ALIGN * _STAGING_ALIGN
    return offsets, off


def staging_region_bytes(world_size: int) -> int:
    """Bytes of mori's staging region; 0 while _staging_buffers is empty.

    A caller that lays out its own arena reserves this much in it, on a 128 B line,
    and names it with ``staging_region=`` on EpDispatchPlan. Reserving it before
    there is anything to put in it is the point: the caller wires the region in
    once and a buffer added to _staging_buffers reaches it without a second edit.
    """
    return staging_layout(world_size)[1]


def _window_flat_layout(win_handle: int, local_ptr: int) -> tuple[int, int]:
    """(base, stride) such that PE p's copy of the window's byte 0 is at
    base + p * stride.

    Read once from the window's device descriptor (ccoWindowDevice in cco.hpp:
    u64 winBase, u32 stride4G, i32 lsaRank), so a kernel reaching a peer's copy
    does no descriptor load of its own. Checked against the local address, which
    cco reports independently: a descriptor laid out differently fails here rather
    than on the GPU, where every peer address it produced would be wrong too.
    """
    import struct

    import torch

    from mori.tensor_utils import from_gpu_ptr

    raw = bytes(from_gpu_ptr(win_handle, (16,), torch.uint8).cpu().tolist())
    win_base, stride4g, lsa_rank = struct.unpack_from("<QIi", raw)
    stride = stride4g << 32
    if stride == 0 or win_base + lsa_rank * stride != local_ptr:
        raise RuntimeError(
            f"staging region: window descriptor (winBase={win_base:#x}, "
            f"stride4G={stride4g}, lsaRank={lsa_rank}) does not locate the local "
            f"copy at {local_ptr:#x}"
        )
    return win_base, stride


def _external_staging(arena, region: str, world_size: int) -> tuple[int, int]:
    """(base, stride) of a staging region the CALLER reserved in its own arena.

    Nothing is allocated or freed here: the arena's owner does both, and its own
    zeroing covers the region. arena.size() is not optional -- overrunning the
    region is not an error the kernel raises, it puts mori's writes in whatever
    region the arena laid down next.
    """
    offset, nbytes = int(arena.offset(region)), staging_region_bytes(world_size)
    if arena.size(region) < nbytes or offset % _STAGING_ALIGN:
        raise RuntimeError(
            f"staging region {region!r} is {arena.size(region)} B at offset {offset}; "
            f"needs {nbytes} B (staging_region_bytes()) on a {_STAGING_ALIGN} B line"
        )
    # local_ptr is the REGION; back out to the window's byte 0, which is what the
    # descriptor's base and stride are relative to, then step back in.
    base, stride = _window_flat_layout(arena.handle, arena.local_ptr(region) - offset)
    return base + offset, stride


class EpDispatchPlan(_EpDispatchPlanBase):
    # The generated signature stays the documentation; this only adds the binding.
    __doc__ = (_EpDispatchPlanBase.__doc__ or "") + (
        "\n\nThe gfx125x dispatch reserves its slots once per destination (aggReserve)"
        "\nunless ``tok_off_ext`` says the slot allocator word lives outside the cco"
        "\nwindow. With it, the plan allocates the state its blocks meet in, zeroed, on"
        "\nthe current device and binds it; zero_state() clears it again."
    )

    def __init__(self, **kwargs):
        arena, region = kwargs.get("arena"), kwargs.pop("staging_region", None)
        self._res_state = None
        # (base, stride) of the caller's staging region, for the buffers
        # _staging_buffers lists to bind from. None when the caller passed no
        # region; empty of buffers either way until one is added there.
        self._staging = None
        super().__init__(**kwargs)
        info = self.info
        if region is not None:
            try:
                self._staging = _external_staging(arena, region, info["worldSize"])
            except BaseException:
                super().close()
                raise
        if not info.get("aggReserve"):
            return
        import torch

        try:
            self._res_state = torch.zeros(
                agg_reserve_state_bytes(info["worldSize"]),
                dtype=torch.uint8,
                device="cuda",
            )
        except BaseException:
            super().close()
            raise
        self.bind(res_state=self._res_state.data_ptr())

    def zero_state(self) -> None:
        """Clear the reservation state. No launch of this plan may be in flight."""
        if self._res_state is not None:
            self._res_state.zero_()

    def close(self) -> None:
        self._res_state = None
        super().close()


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
