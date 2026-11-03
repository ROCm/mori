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
  * the handle's doorbell mode ends the data-path tag (``__deferred`` /
    ``__caller_ordered``) and the flush tag (``__caller_ordered``); see
    ``GDA_DOORBELL_TAG``.
"""

from mori.cco.device.ops import (
    CCO_DEVICE_FUNCTIONS,
    COOP_TAGS,
)

from ._internal import _ffi


def _make(meta):
    return _ffi(
        meta["symbol"],
        meta["args"],
        meta["ret"],
        pure=meta.get("pure", False),
    )


# Every registered function, keyed like CCO_DEVICE_FUNCTIONS; the tables below
# are views of it, so a registered function always has a binding.
ALL = {name: _make(meta) for name, meta in CCO_DEVICE_FUNCTIONS.items()}


def _binding(name):
    return ALL[name]


def _by_tag(family):
    return {
        meta["tag"]: ALL[name]
        for name, meta in CCO_DEVICE_FUNCTIONS.items()
        if meta["family"] == family
    }


# ── monomorphized op tables (keyed by tag) ──
PUT = _by_tag("gda_put")  # "<tc>__<sig><doorbell>"
PUT_VALUE = _by_tag("gda_put_value")
GET = _by_tag("gda_get")  # "<tc><doorbell>"
SIGNAL = {
    f"{c}__{s}": _binding(f"gda_signal_{c}_{s}")
    for c in COOP_TAGS
    for s in ("inc", "add")
}
WAIT_SIGNAL = {c: _binding(f"gda_wait_signal_{c}") for c in COOP_TAGS}
FLUSH = _by_tag("gda_flush")  # "<coop><flush doorbell>"
FLUSH_PEER = _by_tag("gda_flush_peer")
FLUSH_ASYNC = _by_tag("gda_flush_async")
WAIT = {c: _binding(f"gda_wait_{c}") for c in COOP_TAGS}

# ── SDMA ──

SDMA_XFER = {
    f"{op}__{s}": _binding(f"sdma_{op}_{s}")
    for op in ("put", "get")
    # coop tag, plus "_ns" (no-signal / fire-and-forget) variants.
    for s in ("thread", "warp", "block", "thread_ns", "warp_ns", "block_ns")
}

SDMA_XFER2D = {
    f"{op}2d__{s}": _binding(f"sdma_{op}2d_{s}")
    for op in ("put", "get")
    for s in ("thread", "warp", "block", "thread_ns", "warp_ns", "block_ns")
}

SDMA_QUIET = {s: _binding(f"sdma_quiet_{s}") for s in ("thread", "warp", "block")}
SDMA_COMMIT = {s: _binding(f"sdma_commit_{s}") for s in ("thread", "warp", "block")}
cco_sdma_quiet_queue = _binding("sdma_quiet_queue")
# ccoSdma::putPtr / getPtr by coop tag; the last argument is SdmaFlags.
SDMA_PUT_PTR = {s: _binding(f"sdma_put_ptr_{s}") for s in ("thread", "warp", "block")}
SDMA_GET_PTR = {s: _binding(f"sdma_get_ptr_{s}") for s in ("thread", "warp", "block")}
cco_sdma_wait_signal = _binding("sdma_wait_signal")
cco_sdma_put_ptr_rsig_thread = _binding("sdma_put_ptr_rsig_thread")
cco_sdma_signal_slot = _binding("sdma_signal_slot")
# ccoSdma::postWaitMem / postWaitSignal / putBroadcast by coop tag.
SDMA_POST_WAIT_MEM = {
    s: _binding(f"sdma_post_wait_mem_{s}") for s in ("thread", "warp", "block")
}
SDMA_POST_WAIT_SIGNAL = {
    s: _binding(f"sdma_post_wait_signal_{s}") for s in ("thread", "warp", "block")
}
SDMA_PUT_BROADCAST = {
    s: _binding(f"sdma_put_broadcast_{s}") for s in ("thread", "warp", "block")
}
cco_sdma_post_wait_mem_put_ptr_thread = _binding("sdma_post_wait_mem_put_ptr_thread")

# ── axis-free symbols ──
# cco_lsa_ptr(window, peerLsaRank, offset) -> peer's load/store-accessible VA.
cco_lsa_ptr = _binding("lsa_ptr")
# Window geometry: one global_load each, for callers that take it once.
cco_lsa_win_base = _binding("lsa_win_base")
cco_lsa_stride = _binding("lsa_stride")
cco_local_ptr = _binding("local_ptr")
cco_system_fence = _binding("system_fence")

cco_devcomm_rank = _binding("devcomm_rank")
cco_devcomm_world_size = _binding("devcomm_world_size")
cco_devcomm_lsa_rank = _binding("devcomm_lsa_rank")
cco_devcomm_lsa_size = _binding("devcomm_lsa_size")

cco_gda_read_signal = _binding("gda_read_signal")
cco_gda_reset_signal = _binding("gda_reset_signal")
