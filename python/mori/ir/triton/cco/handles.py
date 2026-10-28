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
"""Method-group facades for the Triton CCO device API.

Triton has no FlyDSL-style runtime struct that can retain methods around a
tensor handle.  These classes are therefore compile-time namespaces: the raw
``dev_comm`` / window handle remains the first argument, while method names and
constexpr template axes match ``mori.cco.device.flydsl``.
"""

import triton
import triton.language as tl

from mori.cco.device.ops import (
    CoopScope,
    SdmaFlags,
    SdmaOptFlags,
    SignalOp,
    ThreadMode,
)

from . import ops as raw

_COOP_THREAD = tl.constexpr(CoopScope.THREAD)
_COOP_WARP = tl.constexpr(CoopScope.WARP)
_COOP_BLOCK = tl.constexpr(CoopScope.BLOCK)
_SIGNAL_NONE = tl.constexpr(SignalOp.NONE)
_SIGNAL_INC = tl.constexpr(SignalOp.INC)
_SIGNAL_ADD = tl.constexpr(SignalOp.ADD)
_THREAD_INDEPENDENT = tl.constexpr(ThreadMode.INDEPENDENT)
_THREAD_AGGREGATE = tl.constexpr(ThreadMode.AGGREGATE)
_FLAG_AGGREGATE = tl.constexpr(SdmaFlags.AGGREGATE)
_FLAG_SIGNAL_PER_COPY = tl.constexpr(SdmaFlags.SIGNAL_PER_COPY)
_FLAG_LOCAL_SIGNAL = tl.constexpr(SdmaFlags.LOCAL_SIGNAL)
_FLAG_REMOTE_SIGNAL = tl.constexpr(SdmaFlags.REMOTE_SIGNAL)
_FLAG_SAME_QUEUE = tl.constexpr(SdmaFlags.SAME_QUEUE)


@triton.jit
def _gda_put(
    dev_comm,
    peer,
    dst_win,
    dst_off,
    src_win,
    src_off,
    nbytes,
    ctx: tl.constexpr = 0,
    signal_op: tl.constexpr = _SIGNAL_NONE,
    signal_id=0,
    signal_val=0,
    coop: tl.constexpr = _COOP_THREAD,
    thread_mode: tl.constexpr = _THREAD_INDEPENDENT,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "GDA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    tl.static_assert(
        thread_mode == _THREAD_INDEPENDENT or thread_mode == _THREAD_AGGREGATE,
        "GDA thread_mode must be ThreadMode.INDEPENDENT or AGGREGATE",
    )
    tl.static_assert(
        signal_op == _SIGNAL_NONE
        or signal_op == _SIGNAL_INC
        or signal_op == _SIGNAL_ADD,
        "GDA signal_op must be SignalOp.NONE, INC, or ADD",
    )
    tl.static_assert(
        thread_mode == _THREAD_INDEPENDENT or coop == _COOP_THREAD,
        "GDA aggregate thread mode requires thread cooperative scope",
    )
    if thread_mode == _THREAD_AGGREGATE:
        if signal_op == _SIGNAL_NONE:
            raw.gda_put_at_none(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
        elif signal_op == _SIGNAL_INC:
            raw.gda_put_at_inc(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
        else:
            raw.gda_put_at_add(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
    elif coop == _COOP_THREAD:
        if signal_op == _SIGNAL_NONE:
            raw.gda_put_it_none(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
        elif signal_op == _SIGNAL_INC:
            raw.gda_put_it_inc(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
        else:
            raw.gda_put_it_add(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
    elif coop == _COOP_WARP:
        if signal_op == _SIGNAL_NONE:
            raw.gda_put_iw_none(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
        elif signal_op == _SIGNAL_INC:
            raw.gda_put_iw_inc(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
        else:
            raw.gda_put_iw_add(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
    else:
        if signal_op == _SIGNAL_NONE:
            raw.gda_put_ib_none(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
        elif signal_op == _SIGNAL_INC:
            raw.gda_put_ib_inc(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )
        else:
            raw.gda_put_ib_add(
                dev_comm,
                ctx,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                signal_id,
                signal_val,
            )


@triton.jit
def _gda_put_value(
    dev_comm,
    peer,
    dst_win,
    dst_off,
    value,
    ctx: tl.constexpr = 0,
    signal_op: tl.constexpr = _SIGNAL_NONE,
    signal_id=0,
    signal_val=0,
    coop: tl.constexpr = _COOP_THREAD,
    thread_mode: tl.constexpr = _THREAD_INDEPENDENT,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "GDA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    tl.static_assert(
        thread_mode == _THREAD_INDEPENDENT or thread_mode == _THREAD_AGGREGATE,
        "GDA thread_mode must be ThreadMode.INDEPENDENT or AGGREGATE",
    )
    tl.static_assert(
        signal_op == _SIGNAL_NONE
        or signal_op == _SIGNAL_INC
        or signal_op == _SIGNAL_ADD,
        "GDA signal_op must be SignalOp.NONE, INC, or ADD",
    )
    tl.static_assert(
        thread_mode == _THREAD_INDEPENDENT or coop == _COOP_THREAD,
        "GDA aggregate thread mode requires thread cooperative scope",
    )
    if thread_mode == _THREAD_AGGREGATE:
        if signal_op == _SIGNAL_NONE:
            raw.gda_put_value_at_none(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        elif signal_op == _SIGNAL_INC:
            raw.gda_put_value_at_inc(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        else:
            raw.gda_put_value_at_add(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
    elif coop == _COOP_THREAD:
        if signal_op == _SIGNAL_NONE:
            raw.gda_put_value_it_none(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        elif signal_op == _SIGNAL_INC:
            raw.gda_put_value_it_inc(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        else:
            raw.gda_put_value_it_add(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
    elif coop == _COOP_WARP:
        if signal_op == _SIGNAL_NONE:
            raw.gda_put_value_iw_none(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        elif signal_op == _SIGNAL_INC:
            raw.gda_put_value_iw_inc(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        else:
            raw.gda_put_value_iw_add(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
    else:
        if signal_op == _SIGNAL_NONE:
            raw.gda_put_value_ib_none(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        elif signal_op == _SIGNAL_INC:
            raw.gda_put_value_ib_inc(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        else:
            raw.gda_put_value_ib_add(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )


@triton.jit
def _gda_get(
    dev_comm,
    peer,
    remote_win,
    remote_off,
    local_win,
    local_off,
    nbytes,
    ctx: tl.constexpr = 0,
    coop: tl.constexpr = _COOP_THREAD,
    thread_mode: tl.constexpr = _THREAD_INDEPENDENT,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "GDA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    tl.static_assert(
        thread_mode == _THREAD_INDEPENDENT or thread_mode == _THREAD_AGGREGATE,
        "GDA thread_mode must be ThreadMode.INDEPENDENT or AGGREGATE",
    )
    tl.static_assert(
        thread_mode == _THREAD_INDEPENDENT or coop == _COOP_THREAD,
        "GDA aggregate thread mode requires thread cooperative scope",
    )
    if thread_mode == _THREAD_AGGREGATE:
        raw.gda_get_at(
            dev_comm, ctx, peer, remote_win, remote_off, local_win, local_off, nbytes
        )
    elif coop == _COOP_THREAD:
        raw.gda_get_it(
            dev_comm, ctx, peer, remote_win, remote_off, local_win, local_off, nbytes
        )
    elif coop == _COOP_WARP:
        raw.gda_get_iw(
            dev_comm, ctx, peer, remote_win, remote_off, local_win, local_off, nbytes
        )
    else:
        raw.gda_get_ib(
            dev_comm, ctx, peer, remote_win, remote_off, local_win, local_off, nbytes
        )


@triton.jit
def _gda_signal(
    dev_comm,
    peer,
    ctx: tl.constexpr = 0,
    signal_op: tl.constexpr = _SIGNAL_INC,
    signal_id=0,
    signal_val=0,
    coop: tl.constexpr = _COOP_THREAD,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "GDA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    tl.static_assert(
        signal_op == _SIGNAL_INC or signal_op == _SIGNAL_ADD,
        "GDA signal requires SignalOp.INC or ADD",
    )
    if coop == _COOP_THREAD:
        if signal_op == _SIGNAL_INC:
            raw.gda_signal_thread_inc(dev_comm, ctx, peer, signal_id, signal_val)
        else:
            raw.gda_signal_thread_add(dev_comm, ctx, peer, signal_id, signal_val)
    elif coop == _COOP_WARP:
        if signal_op == _SIGNAL_INC:
            raw.gda_signal_warp_inc(dev_comm, ctx, peer, signal_id, signal_val)
        else:
            raw.gda_signal_warp_add(dev_comm, ctx, peer, signal_id, signal_val)
    else:
        if signal_op == _SIGNAL_INC:
            raw.gda_signal_block_inc(dev_comm, ctx, peer, signal_id, signal_val)
        else:
            raw.gda_signal_block_add(dev_comm, ctx, peer, signal_id, signal_val)


@triton.jit
def _gda_read_signal(
    dev_comm,
    signal_id,
    bits=64,
    ctx: tl.constexpr = 0,
):
    return raw.gda_read_signal(dev_comm, ctx, signal_id, bits)


@triton.jit
def _gda_reset_signal(
    dev_comm,
    signal_id,
    ctx: tl.constexpr = 0,
):
    raw.gda_reset_signal(dev_comm, ctx, signal_id)


@triton.jit
def _gda_wait_signal(
    dev_comm,
    signal_id,
    least,
    bits=64,
    ctx: tl.constexpr = 0,
    coop: tl.constexpr = _COOP_THREAD,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "GDA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    if coop == _COOP_THREAD:
        raw.gda_wait_signal_thread(dev_comm, ctx, signal_id, least, bits)
    elif coop == _COOP_WARP:
        raw.gda_wait_signal_warp(dev_comm, ctx, signal_id, least, bits)
    else:
        raw.gda_wait_signal_block(dev_comm, ctx, signal_id, least, bits)


@triton.jit
def _gda_flush(
    dev_comm,
    ctx: tl.constexpr = 0,
    coop: tl.constexpr = _COOP_WARP,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "GDA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    if coop == _COOP_BLOCK:
        raw.gda_flush_block(dev_comm, ctx)
    else:
        raw.gda_flush_warp(dev_comm, ctx)


@triton.jit
def _gda_flush_peer(
    dev_comm,
    peer,
    ctx: tl.constexpr = 0,
    coop: tl.constexpr = _COOP_WARP,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "GDA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    if coop == _COOP_BLOCK:
        raw.gda_flush_peer_block(dev_comm, ctx, peer)
    else:
        raw.gda_flush_peer_warp(dev_comm, ctx, peer)


# put / get / put_ptr / get_ptr mirror ccoSdma's, and their keywords its template
# arguments (all tl.constexpr): coop, local_signal, remote_signal, aggregate,
# signal_per_copy, same_queue. A signal is a trailing ATOMIC bumping
# signal_slot(<sender lsa rank>, qid): ours for local_signal, the peer's for
# remote_signal. put / get also take `signal`, the older name of local_signal; a
# call that names no signal keeps their old default, a local one. The keywords
# fold into a constant SdmaFlags word, so the wrapper's dispatch compiles away.


@triton.jit
def _sdma_ptr_op(
    op: tl.constexpr,
    dev_comm,
    peer,
    dst,
    src,
    nbytes,
    qid,
    coop: tl.constexpr,
    local_signal: tl.constexpr,
    remote_signal: tl.constexpr,
    aggregate: tl.constexpr,
    signal_per_copy: tl.constexpr,
    same_queue: tl.constexpr,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "SDMA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    tl.static_assert(
        not same_queue or coop == _COOP_THREAD,
        "SDMA same_queue requires CoopScope.THREAD",
    )
    flags: tl.constexpr = (
        (_FLAG_LOCAL_SIGNAL if local_signal else 0)
        | (_FLAG_REMOTE_SIGNAL if remote_signal else 0)
        | (_FLAG_AGGREGATE if aggregate else 0)
        | (_FLAG_SIGNAL_PER_COPY if signal_per_copy else 0)
        | (_FLAG_SAME_QUEUE if same_queue else 0)
    )
    if op == "put":
        if coop == _COOP_THREAD:
            raw.sdma_put_ptr_thread(dev_comm, peer, dst, src, nbytes, qid, flags)
        elif coop == _COOP_WARP:
            raw.sdma_put_ptr_warp(dev_comm, peer, dst, src, nbytes, qid, flags)
        else:
            raw.sdma_put_ptr_block(dev_comm, peer, dst, src, nbytes, qid, flags)
    else:
        if coop == _COOP_THREAD:
            raw.sdma_get_ptr_thread(dev_comm, peer, dst, src, nbytes, qid, flags)
        elif coop == _COOP_WARP:
            raw.sdma_get_ptr_warp(dev_comm, peer, dst, src, nbytes, qid, flags)
        else:
            raw.sdma_get_ptr_block(dev_comm, peer, dst, src, nbytes, qid, flags)


@triton.jit
def _sdma_put_ptr(
    dev_comm,
    peer,
    dst,
    src,
    nbytes,
    qid=0,
    coop: tl.constexpr = _COOP_THREAD,
    local_signal: tl.constexpr = False,
    remote_signal: tl.constexpr = False,
    aggregate: tl.constexpr = False,
    signal_per_copy: tl.constexpr = False,
    same_queue: tl.constexpr = False,
):
    # dst: peer's memory as mapped here (Window.lsa_ptr); src: ours. Unchecked.
    _sdma_ptr_op(
        "put",
        dev_comm,
        peer,
        dst,
        src,
        nbytes,
        qid,
        coop,
        local_signal,
        remote_signal,
        aggregate,
        signal_per_copy,
        same_queue,
    )


@triton.jit
def _sdma_get_ptr(
    dev_comm,
    peer,
    dst,
    src,
    nbytes,
    qid=0,
    coop: tl.constexpr = _COOP_THREAD,
    local_signal: tl.constexpr = False,
    remote_signal: tl.constexpr = False,
    aggregate: tl.constexpr = False,
    signal_per_copy: tl.constexpr = False,
    same_queue: tl.constexpr = False,
):
    # dst: ours; src: peer's memory as mapped here (Window.lsa_ptr). Unchecked.
    _sdma_ptr_op(
        "get",
        dev_comm,
        peer,
        dst,
        src,
        nbytes,
        qid,
        coop,
        local_signal,
        remote_signal,
        aggregate,
        signal_per_copy,
        same_queue,
    )


@triton.jit
def _sdma_put(
    dev_comm,
    peer,
    dst_win,
    dst_off,
    src_win,
    src_off,
    nbytes,
    qid=0,
    coop: tl.constexpr = _COOP_THREAD,
    signal: tl.constexpr = None,
    local_signal: tl.constexpr = None,
    remote_signal: tl.constexpr = None,
    aggregate: tl.constexpr = False,
    signal_per_copy: tl.constexpr = False,
    same_queue: tl.constexpr = False,
):
    tl.static_assert(
        signal is None or local_signal is None,
        "SDMA put: pass signal or local_signal, not both",
    )
    _sdma_ptr_op(
        "put",
        dev_comm,
        peer,
        raw.lsa_ptr(dst_win, peer, dst_off),
        raw.local_ptr(src_win, src_off),
        nbytes,
        qid,
        coop,
        # sdma_signals(): `signal` is local_signal's old name; naming no signal
        # keeps the old default, a local one
        (
            local_signal
            if local_signal is not None
            else (signal if signal is not None else remote_signal is None)
        ),
        remote_signal is not None and remote_signal,
        aggregate,
        signal_per_copy,
        same_queue,
    )


@triton.jit
def _sdma_get(
    dev_comm,
    peer,
    dst_win,
    dst_off,
    src_win,
    src_off,
    nbytes,
    qid=0,
    coop: tl.constexpr = _COOP_THREAD,
    signal: tl.constexpr = None,
    local_signal: tl.constexpr = None,
    remote_signal: tl.constexpr = None,
    aggregate: tl.constexpr = False,
    signal_per_copy: tl.constexpr = False,
    same_queue: tl.constexpr = False,
):
    tl.static_assert(
        signal is None or local_signal is None,
        "SDMA get: pass signal or local_signal, not both",
    )
    _sdma_ptr_op(
        "get",
        dev_comm,
        peer,
        raw.local_ptr(dst_win, dst_off),
        raw.lsa_ptr(src_win, peer, src_off),
        nbytes,
        qid,
        coop,
        # sdma_signals(): `signal` is local_signal's old name; naming no signal
        # keeps the old default, a local one
        (
            local_signal
            if local_signal is not None
            else (signal if signal is not None else remote_signal is None)
        ),
        remote_signal is not None and remote_signal,
        aggregate,
        signal_per_copy,
        same_queue,
    )


@triton.jit
def _sdma_commit(
    dev_comm,
    peer,
    qid=0,
    coop: tl.constexpr = _COOP_THREAD,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "SDMA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    if coop == _COOP_THREAD:
        raw.sdma_commit_thread(dev_comm, peer, qid)
    elif coop == _COOP_WARP:
        raw.sdma_commit_warp(dev_comm, peer, qid)
    else:
        raw.sdma_commit_block(dev_comm, peer, qid)


@triton.jit
def _sdma_quiet(
    dev_comm,
    peer,
    coop: tl.constexpr = _COOP_THREAD,
):
    tl.static_assert(
        coop == _COOP_THREAD or coop == _COOP_WARP or coop == _COOP_BLOCK,
        "SDMA coop must be CoopScope.THREAD, WARP, or BLOCK",
    )
    if coop == _COOP_THREAD:
        raw.sdma_quiet_thread(dev_comm, peer)
    elif coop == _COOP_WARP:
        raw.sdma_quiet_warp(dev_comm, peer)
    else:
        raw.sdma_quiet_block(dev_comm, peer)


class Window:
    """Namespace for operations on a ``ccoWindow_t`` handle."""

    lsa_ptr = staticmethod(raw.lsa_ptr)


class DevComm:
    """Namespace for fields on a device-resident ``ccoDevComm`` handle."""

    rank = staticmethod(raw.devcomm_rank)
    world_size = staticmethod(raw.devcomm_world_size)
    lsa_rank = staticmethod(raw.devcomm_lsa_rank)
    lsa_size = staticmethod(raw.devcomm_lsa_size)


class Gda:
    """Namespace matching the FlyDSL ``Gda`` method names."""

    put = staticmethod(_gda_put)
    put_value = staticmethod(_gda_put_value)
    get = staticmethod(_gda_get)
    signal = staticmethod(_gda_signal)
    read_signal = staticmethod(_gda_read_signal)
    reset_signal = staticmethod(_gda_reset_signal)
    wait_signal = staticmethod(_gda_wait_signal)
    flush = staticmethod(_gda_flush)
    flush_peer = staticmethod(_gda_flush_peer)


class Sdma:
    """Namespace matching the FlyDSL ``Sdma`` method names."""

    put = staticmethod(_sdma_put)
    get = staticmethod(_sdma_get)
    commit = staticmethod(_sdma_commit)
    quiet = staticmethod(_sdma_quiet)
    quiet_queue = staticmethod(raw.sdma_quiet_queue)
    put_ptr = staticmethod(_sdma_put_ptr)
    get_ptr = staticmethod(_sdma_get_ptr)
    wait_signal = staticmethod(raw.sdma_wait_signal)
    signal_slot = staticmethod(raw.sdma_signal_slot)


__all__ = [
    "DevComm",
    "Window",
    "Gda",
    "Sdma",
    "CoopScope",
    "SignalOp",
    "ThreadMode",
    "SdmaOptFlags",
    "SdmaFlags",
]
