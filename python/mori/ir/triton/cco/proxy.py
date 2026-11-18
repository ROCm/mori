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
"""Triton facade for the ccoProxy device API (host-proxy RDMA, ``cco_proxy.hpp``).

Every call names a queue: ``queue`` indexes the DevComm's SPSC pool
(``ProxyPool.SPSC``, ``proxy_spsc_queue_count`` queues) or its MPSC pool
(``ProxyPool.MPSC``, ``proxy_mpsc_queue_count``). Each call constructs a C++
``ccoProxy`` handle on that queue and destroys it at return:

* SPSC: the handle loads the queue's producer state and saves it back after a
  push, so successive calls on one queue continue it, within a kernel and
  across launches. One wavefront must own an SPSC queue: with
  ``coop=CoopScope.BLOCK`` that is wavefront 0 of the program, so give every
  program that runs at the same time its own queue (e.g.
  ``queue=tl.program_id(0)`` with one SPSC queue per program). Out of range
  traps. MPSC queues wrap and may be shared by programs.
* ``coop=CoopScope.BLOCK`` (default): the program issues one op from
  wavefront 0. ``coop=CoopScope.WARP``: every wavefront of the program issues
  its own op with the same arguments, so use it with ``num_warps=1`` (where it
  equals BLOCK): with more warps an SPSC queue gets several producers and is
  corrupted, and an MPSC queue gets the op once per wavefront.
* ``flush`` waits until every op pushed to ``queue`` before it, by any earlier
  call or launch, has completed on every peer (sources reusable). A BLOCK flush
  flushes ``queue`` only, from wavefront 0: it does not cover other queues.

All threads of the program must make the call (Triton scalar code does).
put / put_value / signal fence the program's earlier stores to the NIC first.
``SignalOp.SET`` writes the value with an RDMA write instead of a NIC atomic:
send every set of a slot through one queue, with increasing values and no INC /
ADD to that slot; on a reset slot ``wait_signal(id, v)`` waits for v. Waits
consume what they waited for, as GDA's. Needs a wave64 gfx942 / gfx950 GPU
(the wrappers trap elsewhere, e.g. gfx1250), a MORI build with the ccoProxy
host engine, and a DevComm whose pool is non-empty (an empty pool traps).
"""

import triton
import triton.language as tl

from mori.cco.device.ops import CoopScope, GdaFlags, ProxyPool, SignalOp, Team

from . import ops as raw

_COOP_BLOCK = tl.constexpr(CoopScope.BLOCK)
_SIGNAL_NONE = tl.constexpr(SignalOp.NONE)
_SIGNAL_INC = tl.constexpr(SignalOp.INC)
_POOL_SPSC = tl.constexpr(ProxyPool.SPSC)
_TEAM_WORLD = tl.constexpr(Team.WORLD)
_TEAM_GDA = tl.constexpr(Team.GDA)
_GDA_FLAG_TEAM_GDA = tl.constexpr(GdaFlags.TEAM_GDA)


@triton.jit
def _proxy_put(
    dev_comm,
    queue,
    peer,
    dst_win,
    dst_off,
    src_win,
    src_off,
    nbytes,
    signal_op: tl.constexpr = _SIGNAL_NONE,
    signal_id=0,
    signal_val=0,
    coop: tl.constexpr = _COOP_BLOCK,
    pool: tl.constexpr = _POOL_SPSC,
    team: tl.constexpr = _TEAM_WORLD,
):
    """RDMA write of ``[src_win+src_off, +nbytes)`` to ``peer``'s
    ``[dst_win+dst_off, ...)``, then (same QP, after the data) the signal
    ``signal_op`` (NONE / INC / ADD / SET) on slot ``signal_id`` with
    ``signal_val`` (ADD / SET). ``team`` numbers ``peer`` as for GDA."""
    tl.static_assert(
        team == _TEAM_WORLD or team == _TEAM_GDA,
        "Proxy team must be Team.WORLD or Team.GDA",
    )
    flags: tl.constexpr = _GDA_FLAG_TEAM_GDA if team == _TEAM_GDA else 0
    raw._proxy(
        "put",
        coop,
        signal_op,
        pool,
        dev_comm,
        queue,
        peer,
        dst_win,
        dst_off,
        src_win,
        src_off,
        nbytes,
        signal_id,
        signal_val,
        flags,
    )


@triton.jit
def _proxy_put_value(
    dev_comm,
    queue,
    peer,
    dst_win,
    dst_off,
    value,
    signal_op: tl.constexpr = _SIGNAL_NONE,
    signal_id=0,
    signal_val=0,
    coop: tl.constexpr = _COOP_BLOCK,
    pool: tl.constexpr = _POOL_SPSC,
    team: tl.constexpr = _TEAM_WORLD,
):
    """Inline RDMA write of the 8-byte ``value`` (uint64) to ``peer``'s
    ``[dst_win+dst_off, +8)``, no source buffer, optionally followed by a
    ``SignalOp.SET`` signal as ``put``. signal_op: NONE or SET."""
    tl.static_assert(
        team == _TEAM_WORLD or team == _TEAM_GDA,
        "Proxy team must be Team.WORLD or Team.GDA",
    )
    flags: tl.constexpr = _GDA_FLAG_TEAM_GDA if team == _TEAM_GDA else 0
    raw._proxy(
        "put_value",
        coop,
        signal_op,
        pool,
        dev_comm,
        queue,
        peer,
        dst_win,
        dst_off,
        value,
        signal_id,
        signal_val,
        flags,
    )


@triton.jit
def _proxy_signal(
    dev_comm,
    queue,
    peer,
    signal_op: tl.constexpr = _SIGNAL_INC,
    signal_id=0,
    signal_val=0,
    coop: tl.constexpr = _COOP_BLOCK,
    pool: tl.constexpr = _POOL_SPSC,
    team: tl.constexpr = _TEAM_WORLD,
):
    """Signal only: INC / ADD / SET on ``peer``'s slot ``signal_id``, ordered
    after earlier puts to ``peer`` through the same queue."""
    tl.static_assert(
        team == _TEAM_WORLD or team == _TEAM_GDA,
        "Proxy team must be Team.WORLD or Team.GDA",
    )
    flags: tl.constexpr = _GDA_FLAG_TEAM_GDA if team == _TEAM_GDA else 0
    raw._proxy(
        "signal",
        coop,
        signal_op,
        pool,
        dev_comm,
        queue,
        peer,
        signal_id,
        signal_val,
        flags,
    )


@triton.jit
def _proxy_flush(
    dev_comm,
    queue,
    coop: tl.constexpr = _COOP_BLOCK,
    pool: tl.constexpr = _POOL_SPSC,
):
    """Wait until every op pushed to ``queue`` before this call has completed
    on every peer, then acquire; the sources of earlier puts are reusable.
    BLOCK: wavefront 0 flushes ``queue`` only. Traps once the DevComm failed."""
    raw._proxy("flush", coop, 0, pool, dev_comm, queue)


@triton.jit
def _proxy_flush_timeout(
    dev_comm,
    queue,
    timeout_cycles,
    coop: tl.constexpr = _COOP_BLOCK,
    pool: tl.constexpr = _POOL_SPSC,
):
    """``flush`` that never traps. Returns int32 0 done, 1 timed out after
    ``timeout_cycles`` clock64 cycles (earlier ops may still be in flight; a
    later flush covers them), 2 the DevComm failed; the same in every thread."""
    return raw._proxy("flush_timeout", coop, 0, pool, dev_comm, queue, timeout_cycles)


@triton.jit
def _proxy_wait_signal(
    dev_comm,
    queue,
    signal_id,
    least,
    bits=64,
    coop: tl.constexpr = _COOP_BLOCK,
    pool: tl.constexpr = _POOL_SPSC,
):
    """Wait until local slot ``signal_id`` has gained ``least`` since the
    last wait (consuming it), then acquire. One thread of the group waits
    and consumes once. WARP is valid only with num_warps=1: several
    wavefronts would race on the consumed count; use BLOCK instead. Traps
    once the DevComm failed."""
    raw._proxy("wait_signal", coop, 0, pool, dev_comm, queue, signal_id, least, bits)


@triton.jit
def _proxy_read_signal(
    dev_comm,
    queue,
    signal_id,
    bits=64,
    pool: tl.constexpr = _POOL_SPSC,
):
    """Local slot ``signal_id`` minus what waits consumed (uint64); no
    acquire."""
    return raw._proxy("read_signal", 0, 0, pool, dev_comm, queue, signal_id, bits)


@triton.jit
def _proxy_reset_signal(
    dev_comm,
    queue,
    signal_id,
    pool: tl.constexpr = _POOL_SPSC,
):
    """Zero local slot ``signal_id`` and its consumed count. Only while no
    peer signals it."""
    raw._proxy("reset_signal", 0, 0, pool, dev_comm, queue, signal_id)


@triton.jit
def _proxy_failed(
    dev_comm,
    queue,
    pool: tl.constexpr = _POOL_SPSC,
):
    """int32 1 once the DevComm failed (see ccoDevCommGetError), else 0."""
    return raw._proxy("failed", 0, 0, pool, dev_comm, queue)


class Proxy:
    """Namespace for the ccoProxy device API; see the module docstring for the
    per-call handle, queue ownership and coop rules. Every method takes
    ``(dev_comm, queue, ...)`` and a constexpr ``pool`` (ProxyPool.SPSC by
    default)."""

    put = staticmethod(_proxy_put)
    put_value = staticmethod(_proxy_put_value)
    signal = staticmethod(_proxy_signal)
    flush = staticmethod(_proxy_flush)
    flush_timeout = staticmethod(_proxy_flush_timeout)
    wait_signal = staticmethod(_proxy_wait_signal)
    read_signal = staticmethod(_proxy_read_signal)
    reset_signal = staticmethod(_proxy_reset_signal)
    failed = staticmethod(_proxy_failed)


__all__ = ["Proxy", "ProxyPool"]
