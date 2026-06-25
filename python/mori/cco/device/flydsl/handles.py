# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""OO handles + enums for the cco device API (FlyDSL).

``DevComm`` / ``Window`` / ``Gda`` are method-carrying FlyDSL structs (see
:func:`._internal.cco_struct`) — first-class DSL values that flow through scf
control flow. Build once at kernel entry and reuse::

    dc  = cco.DevComm(dev_comm)   # ccoDevComm device pointer (uint64 handle)
    win = cco.Window(win_handle)  # ccoWindow_t (uint64 handle)
    gda = dc.gda(0)
    if tid == 0:
        gda.put(1, recv, 0, send, 0, n, signal_op=cco.SignalOp.INC, coop=cco.CoopScope.THREAD)
    gda.flush(coop=cco.CoopScope.BLOCK)          # same gda, across the dynamic if

``CoopScope`` / ``SignalOp`` mirror the enums in
``src/cco/device/cco_device_wrapper.cpp`` (passed as int32).
"""

import flydsl.expr as fx

from . import _bindings as raw
from ._internal import cco_struct


class CoopScope:
    """Cooperative-group scope for a GDA op."""

    THREAD = 0
    WARP = 1
    BLOCK = 2


class SignalOp:
    """Remote signal action bundled with a put/get/signal."""

    NONE = 0
    INC = 1
    ADD = 2


def _win(x):
    """Accept a Window handle or a raw uint64 handle."""
    return x.handle if hasattr(x, "handle") else x


@cco_struct
class Window:
    """Handle for a ``ccoWindow_t`` (already a device pointer).

    LSA model: get a peer's load/store-accessible VA with :meth:`lsa_ptr`, then
    operate on it directly in the kernel (buffer_load/store).
    """

    handle: fx.Int64

    def lsa_ptr(self, peer_lsa_rank, offset=0):
        """Peer's LSA-accessible VA inside this window (uint64), for direct load/store."""
        return raw.cco_lsa_ptr(self.handle, peer_lsa_rank, offset)


@cco_struct
class Gda:
    """GDA handle bound to a ccoDevComm device pointer + context index.

    ``ctx`` is a compile-time (Constexpr) field, so the handle carries a single
    IR value (dev_comm) and flows cleanly through scf.if / scf.for.
    """

    dev_comm: fx.Int64
    ctx: fx.Constexpr

    # ── data path ──
    def put(self, peer, dst_win, dst_off, src_win, src_off, nbytes, *,
            signal_op=SignalOp.NONE, signal_id=0, signal_val=0, coop=CoopScope.THREAD):
        raw.cco_gda_put(self.dev_comm, self.ctx, peer, _win(dst_win), dst_off, _win(src_win),
                        src_off, nbytes, signal_op, signal_id, signal_val, coop)

    def put_value(self, peer, dst_win, dst_off, value, *,
                  signal_op=SignalOp.NONE, signal_id=0, signal_val=0, coop=CoopScope.THREAD):
        raw.cco_gda_put_value(self.dev_comm, self.ctx, peer, _win(dst_win), dst_off, value,
                              signal_op, signal_id, signal_val, coop)

    def get(self, peer, remote_win, remote_off, local_win, local_off, nbytes, *,
            coop=CoopScope.THREAD):
        raw.cco_gda_get(self.dev_comm, self.ctx, peer, _win(remote_win), remote_off,
                        _win(local_win), local_off, nbytes, coop)

    # ── signal ──
    def signal(self, peer, *, signal_op=SignalOp.INC, signal_id=0, signal_val=0,
               coop=CoopScope.THREAD):
        raw.cco_gda_signal(self.dev_comm, self.ctx, peer, signal_op, signal_id, signal_val, coop)

    def read_signal(self, signal_id, bits=64):
        return raw.cco_gda_read_signal(self.dev_comm, self.ctx, signal_id, bits)

    def reset_signal(self, signal_id):
        raw.cco_gda_reset_signal(self.dev_comm, self.ctx, signal_id)

    def wait_signal(self, signal_id, least, *, bits=64, coop=CoopScope.THREAD):
        raw.cco_gda_wait_signal(self.dev_comm, self.ctx, signal_id, least, bits, coop)

    # ── completion (>= warp) ──
    def flush(self, *, coop=CoopScope.WARP):
        raw.cco_gda_flush(self.dev_comm, self.ctx, coop)

    def flush_peer(self, peer, *, coop=CoopScope.WARP):
        raw.cco_gda_flush_peer(self.dev_comm, self.ctx, peer, coop)


@cco_struct
class DevComm:
    """Handle for a device-resident ``ccoDevComm``."""

    ptr: fx.Int64

    @property
    def rank(self):
        return raw.cco_devcomm_rank(self.ptr)

    @property
    def world_size(self):
        return raw.cco_devcomm_world_size(self.ptr)

    @property
    def lsa_rank(self):
        return raw.cco_devcomm_lsa_rank(self.ptr)

    @property
    def lsa_size(self):
        return raw.cco_devcomm_lsa_size(self.ptr)

    def gda(self, ctx=0) -> Gda:
        """Build a GDA handle on this devComm for the given (compile-time) context index."""
        return Gda(dev_comm=self.ptr, ctx=ctx)
