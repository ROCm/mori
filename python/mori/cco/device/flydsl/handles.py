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

``coop`` / ``signal_op`` / ``thread_mode`` must be compile-time constants
(``CoopScope`` / ``SignalOp`` / ``ThreadMode`` values): each method selects a
fully-specialized wrapper symbol by name, so the kernel emits one direct call to
one ``ccoGda<P>`` instantiation — no runtime dispatch. ``ThreadMode.AGGREGATE``
is only valid with ``CoopScope.THREAD`` (cco coalesces the warp's lanes itself).
"""

try:
    import flydsl.expr as fx
except ImportError as e:  # optional dependency
    raise ImportError(
        "cco FlyDSL bindings require FlyDSL. Install it with: pip install flydsl"
    ) from e

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


class ThreadMode:
    """How a thread's lanes contribute to one GDA data-path op.

    ``INDEPENDENT`` — each participating thread issues its own transfer.
    ``AGGREGATE``   — the warp's lanes are coalesced into one transfer by cco
                      (requires ``CoopScope.THREAD``; all lanes must enter).
    """

    INDEPENDENT = 0
    AGGREGATE = 1


# Tag tables — must match the wrapper / _bindings symbol names.
_SIG_TAG = {SignalOp.NONE: "none", SignalOp.INC: "inc", SignalOp.ADD: "add"}
_COOP_TAG = {CoopScope.THREAD: "thread", CoopScope.WARP: "warp", CoopScope.BLOCK: "block"}
_TC_TAG = {
    (ThreadMode.INDEPENDENT, CoopScope.THREAD): "it",
    (ThreadMode.INDEPENDENT, CoopScope.WARP): "iw",
    (ThreadMode.INDEPENDENT, CoopScope.BLOCK): "ib",
    (ThreadMode.AGGREGATE, CoopScope.THREAD): "at",
}


def _const(value, name):
    """Require a compile-time constant (the axis selects a wrapper symbol)."""
    if not isinstance(value, int):
        raise TypeError(
            f"cco: {name} must be a compile-time constant (a CoopScope / SignalOp / "
            f"ThreadMode value), not a runtime DSL value — got {type(value).__name__}. "
            "The axis picks a specialized wrapper symbol at trace time."
        )
    return value


def _tc(coop, thread_mode):
    """(ThreadMode, Coop) -> data-path tag; rejects the invalid aggregate combos."""
    _const(coop, "coop")
    _const(thread_mode, "thread_mode")
    try:
        return _TC_TAG[(thread_mode, coop)]
    except KeyError:
        raise ValueError(
            "cco: ThreadMode.AGGREGATE requires coop=CoopScope.THREAD "
            "(cco coalesces the warp's lanes itself)."
        )


def _flush_tag(coop):
    """flush needs >= warp; THREAD/WARP -> warp, BLOCK -> block."""
    return "block" if _const(coop, "coop") == CoopScope.BLOCK else "warp"


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
            signal_op=SignalOp.NONE, signal_id=0, signal_val=0,
            coop=CoopScope.THREAD, thread_mode=ThreadMode.INDEPENDENT):
        sym = raw.PUT[f"{_tc(coop, thread_mode)}__{_SIG_TAG[_const(signal_op, 'signal_op')]}"]
        sym(self.dev_comm, self.ctx, peer, _win(dst_win), dst_off, _win(src_win), src_off,
            nbytes, signal_id, signal_val)

    def put_value(self, peer, dst_win, dst_off, value, *,
                  signal_op=SignalOp.NONE, signal_id=0, signal_val=0,
                  coop=CoopScope.THREAD, thread_mode=ThreadMode.INDEPENDENT):
        sym = raw.PUT_VALUE[f"{_tc(coop, thread_mode)}__{_SIG_TAG[_const(signal_op, 'signal_op')]}"]
        sym(self.dev_comm, self.ctx, peer, _win(dst_win), dst_off, value, signal_id, signal_val)

    def get(self, peer, remote_win, remote_off, local_win, local_off, nbytes, *,
            coop=CoopScope.THREAD, thread_mode=ThreadMode.INDEPENDENT):
        raw.GET[_tc(coop, thread_mode)](
            self.dev_comm, self.ctx, peer, _win(remote_win), remote_off,
            _win(local_win), local_off, nbytes)

    # ── signal ──
    def signal(self, peer, *, signal_op=SignalOp.INC, signal_id=0, signal_val=0,
               coop=CoopScope.THREAD):
        _const(signal_op, "signal_op")
        _const(coop, "coop")
        if signal_op == SignalOp.NONE:
            raise ValueError("cco: signal() requires signal_op INC or ADD")
        raw.SIGNAL[f"{_COOP_TAG[coop]}__{_SIG_TAG[signal_op]}"](
            self.dev_comm, self.ctx, peer, signal_id, signal_val)

    def read_signal(self, signal_id, bits=64):
        return raw.cco_gda_read_signal(self.dev_comm, self.ctx, signal_id, bits)

    def reset_signal(self, signal_id):
        raw.cco_gda_reset_signal(self.dev_comm, self.ctx, signal_id)

    def wait_signal(self, signal_id, least, *, bits=64, coop=CoopScope.THREAD):
        raw.WAIT_SIGNAL[_COOP_TAG[_const(coop, "coop")]](
            self.dev_comm, self.ctx, signal_id, least, bits)

    # ── completion (>= warp; THREAD coop maps to warp) ──
    def flush(self, *, coop=CoopScope.WARP):
        raw.FLUSH[_flush_tag(coop)](self.dev_comm, self.ctx)

    def flush_peer(self, peer, *, coop=CoopScope.WARP):
        raw.FLUSH_PEER[_flush_tag(coop)](self.dev_comm, self.ctx, peer)


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
