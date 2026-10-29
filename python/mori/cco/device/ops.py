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
"""Framework-independent ABI metadata for the CCO device bitcode.

The entries in :data:`CCO_DEVICE_FUNCTIONS` map friendly Python names to the
monomorphized ``extern "C"`` symbols emitted by
``src/cco/device/cco_device_wrapper.cpp``.  FlyDSL and Triton both consume this
table so the scalar ABI has one source of truth.

All side-effect-only wrappers return ``int32`` status 0.  Triton's
``extern_elementwise`` requires a concrete return element type, including when
an argument is a block value.
"""

from __future__ import annotations


class CoopScope:
    """Cooperative-group scope used by monomorphized CCO wrappers."""

    THREAD = 0
    WARP = 1
    BLOCK = 2


class SignalOp:
    """Remote action attached to a GDA operation."""

    NONE = 0
    INC = 1
    ADD = 2


class ThreadMode:
    """How participating lanes contribute to a GDA data-path operation.

    ``INDEPENDENT`` makes each participating thread issue its own transfer.
    ``AGGREGATE`` coalesces the warp's lanes into one transfer and is valid only
    with ``CoopScope.THREAD``; every lane in the warp must enter the operation.
    """

    INDEPENDENT = 0
    AGGREGATE = 1


class SdmaOptFlags:
    """C++ ``ccoSdmaOptFlags``. The window put/get wrappers (``cco_sdma_put__<tag>``
    and friends) honour AGGREGATE only; :class:`SdmaFlags` carries both."""

    DEFAULT = 0
    AGGREGATE = 1
    SIGNAL_PER_COPY = 2


class SdmaFlags:
    """The ``flags`` word of ``cco_sdma_{put,get}_ptr__<coop>``: every template axis
    of ``ccoSdma::putPtr`` / ``getPtr`` besides Coop. Bits 0-1 are
    :class:`SdmaOptFlags`. Pass a compile-time constant: the wrapper is inlined
    and its dispatch on ``flags`` then folds to one instantiation."""

    AGGREGATE = SdmaOptFlags.AGGREGATE
    SIGNAL_PER_COPY = SdmaOptFlags.SIGNAL_PER_COPY
    LOCAL_SIGNAL = 1 << 2
    REMOTE_SIGNAL = 1 << 3
    SAME_QUEUE = 1 << 4


class SdmaCmp:
    """C++ ``ccoSdmaCmp``: the comparison of an engine-side wait
    (``Sdma.post_wait_mem``), applied to ``(*flag & mask)`` against ``value``."""

    ALWAYS = 0
    LT = 1
    LE = 2
    EQ = 3
    NE = 4
    GE = 5
    GT = 6


def sdma_flags(
    coop,
    *,
    local_signal=False,
    remote_signal=False,
    aggregate=False,
    signal_per_copy=False,
    same_queue=False,
):
    """:class:`SdmaFlags` for one put/get; the arguments are C++'s template ones."""
    if same_queue and coop != CoopScope.THREAD:
        raise ValueError("cco: same_queue requires CoopScope.THREAD")
    return (
        (SdmaFlags.LOCAL_SIGNAL if local_signal else 0)
        | (SdmaFlags.REMOTE_SIGNAL if remote_signal else 0)
        | (SdmaFlags.AGGREGATE if aggregate else 0)
        | (SdmaFlags.SIGNAL_PER_COPY if signal_per_copy else 0)
        | (SdmaFlags.SAME_QUEUE if same_queue else 0)
    )


def sdma_signals(signal, local_signal, remote_signal):
    """(local_signal, remote_signal) of a window put/get. ``signal`` is the old
    name of ``local_signal``; a call naming neither signal keeps the old
    default, a local one."""
    if signal is not None and local_signal is not None:
        raise TypeError("cco: pass signal or local_signal, not both")
    if local_signal is None:
        local_signal = signal if signal is not None else remote_signal is None
    return bool(local_signal), bool(remote_signal)


SIGNAL_TAG = {
    SignalOp.NONE: "none",
    SignalOp.INC: "inc",
    SignalOp.ADD: "add",
}
COOP_TAG = {
    CoopScope.THREAD: "thread",
    CoopScope.WARP: "warp",
    CoopScope.BLOCK: "block",
}
DATA_PATH_TAG = {
    (ThreadMode.INDEPENDENT, CoopScope.THREAD): "it",
    (ThreadMode.INDEPENDENT, CoopScope.WARP): "iw",
    (ThreadMode.INDEPENDENT, CoopScope.BLOCK): "ib",
    (ThreadMode.AGGREGATE, CoopScope.THREAD): "at",
}

GDA_DATA_TAGS = tuple(DATA_PATH_TAG.values())
GDA_SIGNAL_TAGS = tuple(SIGNAL_TAG.values())
COOP_TAGS = tuple(COOP_TAG.values())

_U64 = "uint64"
_I32 = "int32"

CCO_DEVICE_FUNCTIONS: dict[str, dict] = {}


def _add(
    name: str,
    symbol: str,
    args: list[str],
    ret: str,
    *,
    pure: bool = False,
    family: str,
    tag: str | None = None,
) -> None:
    CCO_DEVICE_FUNCTIONS[name] = {
        "symbol": symbol,
        "args": args,
        "ret": ret,
        "pure": pure,
        "family": family,
        "tag": tag,
    }


# Axis-free LSA and DevComm queries.
_add(
    "lsa_ptr",
    "cco_lsa_ptr",
    [_U64, _I32, _U64],
    _U64,
    pure=True,
    family="lsa",
)
# Window geometry, for callers that need many addresses from one window.
# Both read through an address_space(1) pointer so the access is a global_load
# (vmcnt only) rather than the flat_load a generic pointer forces (vmcnt and
# lgkmcnt). See cco_lsa_win_base in the device wrapper.
_add("lsa_win_base", "cco_lsa_win_base", [_U64], _U64, pure=True, family="lsa")
_add("lsa_stride", "cco_lsa_stride", [_U64], _U64, pure=True, family="lsa")
_add("local_ptr", "cco_local_ptr", [_U64, _U64], _U64, pure=True, family="lsa")

# Benchmark-only publication primitive. It is not a cooperative barrier:
# leaderOnly=1 fences thread 0 only, so producers must synchronize first and
# must not use it as a substitute for a per-lane release fence.
_add(
    "system_fence",
    "cco_system_fence",
    [_I32],
    _I32,
    family="sync",
)
for _field in ("rank", "world_size", "lsa_rank", "lsa_size"):
    _add(
        f"devcomm_{_field}",
        f"cco_devcomm_{_field}",
        [_U64],
        _I32,
        pure=True,
        family="devcomm",
    )

# SDMA put/get variants.
_sdma_xfer_args = [_U64, _I32, _U64, _U64, _U64, _U64, _U64, _I32, _I32]
for _op in ("put", "get"):
    for _tag in (
        "thread",
        "warp",
        "block",
        "thread_ns",
        "warp_ns",
        "block_ns",
    ):
        _add(
            f"sdma_{_op}_{_tag}",
            f"cco_sdma_{_op}__{_tag}",
            _sdma_xfer_args,
            _I32,
            family=f"sdma_{_op}",
            tag=_tag,
        )

# SDMA 2D put/get: (devComm, peer, dstWin, dstOff, dstPitch, srcWin, srcOff,
# srcPitch, widthBytes, height, qid, flags).
_sdma_xfer2d_args = [
    _U64,
    _I32,
    _U64,
    _U64,
    _I32,
    _U64,
    _U64,
    _I32,
    _I32,
    _I32,
    _I32,
    _I32,
]
for _op in ("put", "get"):
    for _tag in (
        "thread",
        "warp",
        "block",
        "thread_ns",
        "warp_ns",
        "block_ns",
    ):
        _add(
            f"sdma_{_op}2d_{_tag}",
            f"cco_sdma_{_op}2d__{_tag}",
            _sdma_xfer2d_args,
            _I32,
            family=f"sdma_{_op}2d",
            tag=_tag,
        )

for _tag in COOP_TAGS:
    _add(
        f"sdma_quiet_{_tag}",
        f"cco_sdma_quiet__{_tag}",
        [_U64, _I32],
        _I32,
        family="sdma_quiet",
        tag=_tag,
    )
    _add(
        f"sdma_commit_{_tag}",
        f"cco_sdma_commit__{_tag}",
        [_U64, _I32, _I32],
        _I32,
        family="sdma_commit",
        tag=_tag,
    )
_add(
    "sdma_quiet_queue",
    "cco_sdma_quiet_queue",
    [_U64, _I32, _I32],
    _I32,
    family="sdma_quiet_queue",
)
# ccoSdma::putPtr / getPtr; the last argument is SdmaFlags.
for _op in ("put", "get"):
    for _tag in COOP_TAGS:
        _add(
            f"sdma_{_op}_ptr_{_tag}",
            f"cco_sdma_{_op}_ptr__{_tag}",
            [_U64, _I32, _U64, _U64, _U64, _I32, _I32],
            _I32,
            family=f"sdma_{_op}_ptr",
            tag=_tag,
        )
_add(
    "sdma_wait_signal",
    "cco_sdma_wait_signal",
    [_U64, _I32, _I32, _U64],
    _I32,
    family="sdma_wait_signal",
)
# Older spelling of sdma_put_ptr_thread with SdmaFlags.REMOTE_SIGNAL.
_add(
    "sdma_put_ptr_rsig_thread",
    "cco_sdma_put_ptr_rsig__thread",
    [_U64, _I32, _U64, _U64, _U64, _I32],
    _I32,
    family="sdma_put_ptr_rsig",
    tag="thread",
)
_add(
    "sdma_signal_slot",
    "cco_sdma_signal_slot",
    [_U64, _I32, _I32],
    _U64,
    pure=True,
    family="sdma_signal_slot",
)

# Engine-side wait and broadcast. `flags` is the SdmaFlags subset the C++ call
# takes: AGGREGATE for the waits; AGGREGATE | LOCAL_SIGNAL | REMOTE_SIGNAL for
# the broadcast. value / mask / expected are 32-bit quantities in a uint64.
for _tag in COOP_TAGS:
    # (devComm, peer, qid, win, off, value, cmp, mask, flags)
    _add(
        f"sdma_post_wait_mem_{_tag}",
        f"cco_sdma_post_wait_mem__{_tag}",
        [_U64, _I32, _I32, _U64, _U64, _U64, _I32, _U64, _I32],
        _I32,
        family="sdma_post_wait_mem",
        tag=_tag,
    )
    # (devComm, peer, qid, srcLsaRank, sigQid, expected, flags)
    _add(
        f"sdma_post_wait_signal_{_tag}",
        f"cco_sdma_post_wait_signal__{_tag}",
        [_U64, _I32, _I32, _I32, _I32, _U64, _I32],
        _I32,
        family="sdma_post_wait_signal",
        tag=_tag,
    )
    # (devComm, peer, dstWin, dstOff, peer2, dstWin2, dstOff2, srcWin, srcOff,
    #  nbytes, qid, flags)
    _add(
        f"sdma_put_broadcast_{_tag}",
        f"cco_sdma_put_broadcast__{_tag}",
        [_U64, _I32, _U64, _U64, _I32, _U64, _U64, _U64, _U64, _U64, _I32, _I32],
        _I32,
        family="sdma_put_broadcast",
        tag=_tag,
    )

# GDA data path.
_gda_put_args = [
    _U64,
    _I32,
    _I32,
    _U64,
    _U64,
    _U64,
    _U64,
    _U64,
    _I32,
    _U64,
]
_gda_put_value_args = [_U64, _I32, _I32, _U64, _U64, _U64, _I32, _U64]
_gda_get_args = [_U64, _I32, _I32, _U64, _U64, _U64, _U64, _U64]

for _tc in GDA_DATA_TAGS:
    for _sig in GDA_SIGNAL_TAGS:
        _tag = f"{_tc}__{_sig}"
        _add(
            f"gda_put_{_tc}_{_sig}",
            f"cco_gda_put__{_tag}",
            _gda_put_args,
            _I32,
            family="gda_put",
            tag=_tag,
        )
        _add(
            f"gda_put_value_{_tc}_{_sig}",
            f"cco_gda_put_value__{_tag}",
            _gda_put_value_args,
            _I32,
            family="gda_put_value",
            tag=_tag,
        )
    _add(
        f"gda_get_{_tc}",
        f"cco_gda_get__{_tc}",
        _gda_get_args,
        _I32,
        family="gda_get",
        tag=_tc,
    )

for _coop in COOP_TAGS:
    for _sig in ("inc", "add"):
        _tag = f"{_coop}__{_sig}"
        _add(
            f"gda_signal_{_coop}_{_sig}",
            f"cco_gda_signal__{_tag}",
            [_U64, _I32, _I32, _I32, _U64],
            _I32,
            family="gda_signal",
            tag=_tag,
        )
    _add(
        f"gda_wait_signal_{_coop}",
        f"cco_gda_wait_signal__{_coop}",
        [_U64, _I32, _I32, _U64, _I32],
        _I32,
        family="gda_wait_signal",
        tag=_coop,
    )

_add(
    "gda_read_signal",
    "cco_gda_read_signal",
    [_U64, _I32, _I32, _I32],
    _U64,
    family="gda_read_signal",
)
_add(
    "gda_reset_signal",
    "cco_gda_reset_signal",
    [_U64, _I32, _I32],
    _I32,
    family="gda_reset_signal",
)

for _coop in ("warp", "block"):
    _add(
        f"gda_flush_{_coop}",
        f"cco_gda_flush__{_coop}",
        [_U64, _I32],
        _I32,
        family="gda_flush",
        tag=_coop,
    )
    _add(
        f"gda_flush_peer_{_coop}",
        f"cco_gda_flush_peer__{_coop}",
        [_U64, _I32, _I32],
        _I32,
        family="gda_flush_peer",
        tag=_coop,
    )

CCO_DEVICE_FUNCTIONS_BY_SYMBOL = {
    meta["symbol"]: (name, meta) for name, meta in CCO_DEVICE_FUNCTIONS.items()
}

__all__ = [
    "CCO_DEVICE_FUNCTIONS",
    "CCO_DEVICE_FUNCTIONS_BY_SYMBOL",
    "CoopScope",
    "SignalOp",
    "ThreadMode",
    "SdmaOptFlags",
    "SdmaFlags",
    "SdmaCmp",
    "sdma_flags",
    "sdma_signals",
    "SIGNAL_TAG",
    "COOP_TAG",
    "DATA_PATH_TAG",
    "GDA_DATA_TAGS",
    "GDA_SIGNAL_TAGS",
    "COOP_TAGS",
]
