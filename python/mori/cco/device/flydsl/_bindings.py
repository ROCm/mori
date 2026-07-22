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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""FlyDSL FFI prototypes for ``src/cco/device/cco_device_wrapper.cpp``.

The wrapper MONOMORPHIZES each template axis into a distinct ``extern "C"``
symbol (one per valid Coop / ThreadMode / RemoteAction combination), so there is
no runtime dispatch: the OO handles in :mod:`.handles` pick the symbol by name
from the (compile-time) coop/thread_mode/signal_op and emit one direct call.

All arguments are scalars (FlyDSL FFI is scalar-only): handles (``ccoDevComm*`` /
``ccoWindow_t``) are ``uint64`` intptrs; signal id/value are ``int32`` / ``uint64``.

Symbol tags (must match the wrapper):
  * data path (``put`` / ``put_value`` / ``get``): ``(ThreadMode, Coop)`` tag —
    ``it`` indep+thread, ``iw`` indep+warp, ``ib`` indep+block, ``at`` aggr+thread
    (aggregate is only valid with thread coop).
  * ``signal`` / ``wait`` / ``flush``: coop-only tag ``thread`` / ``warp`` / ``block``.
  * ``put`` / ``put_value`` / ``signal`` also carry a signal-op tag
    ``none`` / ``inc`` / ``add`` (``signal`` has no ``none``).
"""

from ._internal import _ffi

_U64, _I32 = "uint64", "int32"

# (devComm, ctx, peer, dstWin, dstOff, srcWin, srcOff, bytes, signalId, signalVal)
_PUT_ARGS = [_U64, _I32, _I32, _U64, _U64, _U64, _U64, _U64, _I32, _U64]
# (devComm, ctx, peer, dstWin, dstOff, value, signalId, signalVal)
_PUTV_ARGS = [_U64, _I32, _I32, _U64, _U64, _U64, _I32, _U64]
# (devComm, ctx, peer, remoteWin, remoteOff, localWin, localOff, bytes)
_GET_ARGS = [_U64, _I32, _I32, _U64, _U64, _U64, _U64, _U64]
# (devComm, ctx, peer, signalId, signalVal)
_SIGNAL_ARGS = [_U64, _I32, _I32, _I32, _U64]
# (devComm, ctx, signalId, least, bits)
_WAIT_ARGS = [_U64, _I32, _I32, _U64, _I32]

_TC = ("it", "iw", "ib", "at")  # data-path (ThreadMode, Coop) tags
_SIG = ("none", "inc", "add")  # remote-action tags
_COOP = ("thread", "warp", "block")  # coop-only tags

# ── monomorphized op tables (keyed by tag) ──
PUT = {
    f"{tc}__{s}": _ffi(f"cco_gda_put__{tc}__{s}", _PUT_ARGS, "void")
    for tc in _TC
    for s in _SIG
}
PUT_VALUE = {
    f"{tc}__{s}": _ffi(f"cco_gda_put_value__{tc}__{s}", _PUTV_ARGS, "void")
    for tc in _TC
    for s in _SIG
}
GET = {tc: _ffi(f"cco_gda_get__{tc}", _GET_ARGS, "void") for tc in _TC}
SIGNAL = {
    f"{c}__{s}": _ffi(f"cco_gda_signal__{c}__{s}", _SIGNAL_ARGS, "void")
    for c in _COOP
    for s in ("inc", "add")
}
WAIT_SIGNAL = {c: _ffi(f"cco_gda_wait_signal__{c}", _WAIT_ARGS, "void") for c in _COOP}
FLUSH = {
    c: _ffi(f"cco_gda_flush__{c}", [_U64, _I32], "void") for c in ("warp", "block")
}
FLUSH_PEER = {
    c: _ffi(f"cco_gda_flush_peer__{c}", [_U64, _I32, _I32], "void")
    for c in ("warp", "block")
}

# ── SDMA ──

# (devComm, peer, dstWin, dstOff, srcWin, srcOff, nbytes, queueId, optFlags)
_SDMA_XFER_ARGS = [_U64, _I32, _U64, _U64, _U64, _U64, _U64, _I32, _I32]

SDMA_XFER = {
    f"{op}__{s}": _ffi(f"cco_sdma_{op}__{s}", _SDMA_XFER_ARGS, "void")
    for op in ("put", "get")
    # coop tag, plus "_ns" (no-signal / fire-and-forget) variants.
    for s in ("thread", "warp", "block", "thread_ns", "warp_ns", "block_ns")
}

# quiet: (devComm, peer)
SDMA_QUIET = {
    s: _ffi(f"cco_sdma_quiet__{s}", [_U64, _I32], "void")
    for s in ("thread", "warp", "block")
}
# commit: (devComm, peer, queueId)
SDMA_COMMIT = {
    s: _ffi(f"cco_sdma_commit__{s}", [_U64, _I32, _I32], "void")
    for s in ("thread", "warp", "block")
}
# quiet_queue: (devComm, peer, queueId)
cco_sdma_quiet_queue = _ffi("cco_sdma_quiet_queue", [_U64, _I32, _I32], "void")

# ── axis-free symbols ──
# cco_lsa_ptr(window, peerLsaRank, offset) -> peer's load/store-accessible VA.
cco_lsa_ptr = _ffi("cco_lsa_ptr", [_U64, _I32, _U64], _U64, pure=True)

cco_devcomm_rank = _ffi("cco_devcomm_rank", [_U64], _I32, pure=True)
cco_devcomm_world_size = _ffi("cco_devcomm_world_size", [_U64], _I32, pure=True)
cco_devcomm_lsa_rank = _ffi("cco_devcomm_lsa_rank", [_U64], _I32, pure=True)
cco_devcomm_lsa_size = _ffi("cco_devcomm_lsa_size", [_U64], _I32, pure=True)

cco_gda_read_signal = _ffi("cco_gda_read_signal", [_U64, _I32, _I32, _I32], _U64)
cco_gda_reset_signal = _ffi("cco_gda_reset_signal", [_U64, _I32, _I32], "void")
