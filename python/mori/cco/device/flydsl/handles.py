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

from mori.cco.device.ops import (
    COOP_TAG as _COOP_TAG,
    DATA_PATH_TAG as _TC_TAG,
    SIGNAL_TAG as _SIG_TAG,
    CoopScope,
    SignalOp,
    ThreadMode,
)

from . import _bindings as raw
from ._internal import cco_struct


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


class CachedWindow:
    """A :class:`Window` that reads its geometry once, at construction.

    Drop-in for ``Window`` at the top of a kernel::

        w = cco.CachedWindow(win)       # instead of cco.Window(win)

    and every ``w.lsa_ptr(...)`` after it becomes arithmetic instead of an
    extern call. Nothing else about the kernel changes.

    ``cco_lsa_ptr`` is ``winBase + peer*stride + offset`` and loads *both*
    fields out of the window descriptor on every call. It is opaque to FlyDSL's
    tracer, and although it is ``always_inline`` -- so LLVM does see the loads --
    a kernel that stores through addresses derived from that same base gives
    LLVM no way to prove they are not clobbered, so it reloads them.

    The two reads here go through ``cco_lsa_win_base`` / ``cco_lsa_stride``,
    which take an ``address_space(1)`` pointer so each is a single
    ``global_load``. ``cco_lsa_ptr`` casts to a *generic* pointer, and a generic
    access has to be a ``flat_load`` -- the compiler cannot rule out LDS -- so it
    counts against ``lgkmcnt`` as well as ``vmcnt``, giving any following
    ``s_waitcnt`` one more counter to wait on.

    **What it is worth depends on how often the kernel builds an address, not on
    how many call sites it has.** A/B at ``m=16384 n=7168 k=2048`` on 8x MI355X,
    max_rank_time_us, three alternating repeats::

        mode                Window                    CachedWindow
        split-lsa    1264.69 1264.23 1264.85    1250.25 1256.85 1254.53   -10.7us
        fused-sdma   1110.45 1109.77 1112.85    1110.53 1113.69 1112.61     null

    ``split-lsa`` wins 0.85%, against a Window spread of 0.6us across its three
    runs. Its ``ar_1stage``/``ar_2stage`` build nine peer addresses in *every
    block* of a short kernel, so the extern calls are a real fraction of it.
    ``fused-sdma`` gains nothing: its ``lsa_ptr`` calls are per kernel launch,
    once, against a body that runs for a millisecond -- and the same was true of
    ``kernels_sdma``, where converting all 21 sites moved nothing.

    So: count address constructions per launch, not ``grep -c lsa_ptr``.

    Measure A/B in one session if you revisit this. Absolute numbers on this box
    move ~4% between mornings, across every configuration at once, which is
    several times the effect being looked for.

    Unlike :class:`Window` this is a plain Python object, not a ``cco_struct``,
    so it cannot cross an ``scf.if``/``scf.for`` boundary -- and neither would a
    ``cco_struct`` version, because FlyDSL captures every variable a dynamic
    ``if`` body reads as state and requires each to be a *single* MLIR value.
    ``Window`` qualifies only by having one field; a three-field struct fails the
    same check ("state variable 'w' is list, not an MLIR Value").

    Usually the way out is neither: compute the addresses *before* the branch and
    let the ``if`` capture plain ``Int64``. That works whenever the offsets do
    not depend on the branch, which in practice they rarely do.
    """

    __slots__ = ("handle", "base", "stride")

    def __init__(self, win):
        self.handle = fx.Int64(win)
        self.base = fx.Int64(raw.cco_lsa_win_base(self.handle))
        self.stride = fx.Int64(raw.cco_lsa_stride(self.handle))

    def lsa_ptr(self, peer_lsa_rank, offset=0):
        """Same contract as :meth:`Window.lsa_ptr`, with no extern call."""
        return self.base + fx.Int64(peer_lsa_rank) * self.stride + fx.Int64(offset)


@cco_struct
class Gda:
    """GDA handle bound to a ccoDevComm device pointer + context index.

    ``ctx`` is a compile-time (Constexpr) field, so the handle carries a single
    IR value (dev_comm) and flows cleanly through scf.if / scf.for.
    """

    dev_comm: fx.Int64
    ctx: fx.Constexpr

    # ── data path ──
    def put(
        self,
        peer,
        dst_win,
        dst_off,
        src_win,
        src_off,
        nbytes,
        *,
        signal_op=SignalOp.NONE,
        signal_id=0,
        signal_val=0,
        coop=CoopScope.THREAD,
        thread_mode=ThreadMode.INDEPENDENT,
    ):
        sym = raw.PUT[
            f"{_tc(coop, thread_mode)}__{_SIG_TAG[_const(signal_op, 'signal_op')]}"
        ]
        sym(
            self.dev_comm,
            self.ctx,
            peer,
            _win(dst_win),
            dst_off,
            _win(src_win),
            src_off,
            nbytes,
            signal_id,
            signal_val,
        )

    def put_value(
        self,
        peer,
        dst_win,
        dst_off,
        value,
        *,
        signal_op=SignalOp.NONE,
        signal_id=0,
        signal_val=0,
        coop=CoopScope.THREAD,
        thread_mode=ThreadMode.INDEPENDENT,
    ):
        sym = raw.PUT_VALUE[
            f"{_tc(coop, thread_mode)}__{_SIG_TAG[_const(signal_op, 'signal_op')]}"
        ]
        sym(
            self.dev_comm,
            self.ctx,
            peer,
            _win(dst_win),
            dst_off,
            value,
            signal_id,
            signal_val,
        )

    def get(
        self,
        peer,
        remote_win,
        remote_off,
        local_win,
        local_off,
        nbytes,
        *,
        coop=CoopScope.THREAD,
        thread_mode=ThreadMode.INDEPENDENT,
    ):
        raw.GET[_tc(coop, thread_mode)](
            self.dev_comm,
            self.ctx,
            peer,
            _win(remote_win),
            remote_off,
            _win(local_win),
            local_off,
            nbytes,
        )

    # ── signal ──
    def signal(
        self,
        peer,
        *,
        signal_op=SignalOp.INC,
        signal_id=0,
        signal_val=0,
        coop=CoopScope.THREAD,
    ):
        _const(signal_op, "signal_op")
        _const(coop, "coop")
        if signal_op == SignalOp.NONE:
            raise ValueError("cco: signal() requires signal_op INC or ADD")
        raw.SIGNAL[f"{_COOP_TAG[coop]}__{_SIG_TAG[signal_op]}"](
            self.dev_comm, self.ctx, peer, signal_id, signal_val
        )

    def read_signal(self, signal_id, bits=64):
        return raw.cco_gda_read_signal(self.dev_comm, self.ctx, signal_id, bits)

    def reset_signal(self, signal_id):
        raw.cco_gda_reset_signal(self.dev_comm, self.ctx, signal_id)

    def wait_signal(self, signal_id, least, *, bits=64, coop=CoopScope.THREAD):
        raw.WAIT_SIGNAL[_COOP_TAG[_const(coop, "coop")]](
            self.dev_comm, self.ctx, signal_id, least, bits
        )

    # ── completion (>= warp; THREAD coop maps to warp) ──
    def flush(self, *, coop=CoopScope.WARP):
        raw.FLUSH[_flush_tag(coop)](self.dev_comm, self.ctx)

    def flush_peer(self, peer, *, coop=CoopScope.WARP):
        raw.FLUSH_PEER[_flush_tag(coop)](self.dev_comm, self.ctx, peer)


@cco_struct
class Sdma:
    dev_comm: fx.Int64

    def _xfer(
        self,
        op,
        peer,
        dst_win,
        dst_off,
        src_win,
        src_off,
        nbytes,
        qid,
        coop,
        signal,
        aggregate,
    ):
        # coop/signal are compile-time. "_ns" (no-signal) only drops the trailing
        # ATOMIC; quiet()/quietQueue() drain the queue's read pointer and work on
        # it either way (cco.hpp:1707). aggregate skips the doorbell — call commit().
        tag = _COOP_TAG[_const(coop, "coop")]
        if not signal:
            tag += "_ns"
        flags = 1 if aggregate else 0
        raw.SDMA_XFER[f"{op}__{tag}"](
            self.dev_comm,
            peer,
            _win(dst_win),
            dst_off,
            _win(src_win),
            src_off,
            nbytes,
            qid,
            flags,
        )

    def put(
        self,
        peer,
        dst_win,
        dst_off,
        src_win,
        src_off,
        nbytes,
        qid,
        *,
        coop=CoopScope.THREAD,
        signal=True,
        aggregate=False,
    ):
        self._xfer(
            "put",
            peer,
            dst_win,
            dst_off,
            src_win,
            src_off,
            nbytes,
            qid,
            coop,
            signal,
            aggregate,
        )

    def get(
        self,
        peer,
        dst_win,
        dst_off,
        src_win,
        src_off,
        nbytes,
        qid,
        *,
        coop=CoopScope.THREAD,
        signal=True,
        aggregate=False,
    ):
        self._xfer(
            "get",
            peer,
            dst_win,
            dst_off,
            src_win,
            src_off,
            nbytes,
            qid,
            coop,
            signal,
            aggregate,
        )

    def commit(self, peer, qid=0, *, coop=CoopScope.THREAD):
        """Ring the doorbell for aggregate=True ops (thread: queue `qid`; warp/block: all)."""
        raw.SDMA_COMMIT[_COOP_TAG[_const(coop, "coop")]](self.dev_comm, peer, qid)

    def quiet(self, peer, *, coop=CoopScope.THREAD):
        """Wait for all outstanding SDMA ops to `peer` across every queue."""
        raw.SDMA_QUIET[_COOP_TAG[_const(coop, "coop")]](self.dev_comm, peer)

    def quiet_queue(self, peer, qid):
        """Wait on a single (peer, queueId) queue only."""
        raw.cco_sdma_quiet_queue(self.dev_comm, peer, qid)


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

    def sdma(self) -> Sdma:
        """Build an SDMA handle on this devComm (LSA copy-engine put/get)."""
        return Sdma(dev_comm=self.ptr)
