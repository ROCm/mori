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

from mori.cco.device.ops import CoopScope, SignalOp, ThreadMode, SdmaOptFlags

from . import ops as raw


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
    signal_op: tl.constexpr = 0,
    signal_id=0,
    signal_val=0,
    coop: tl.constexpr = 0,
    thread_mode: tl.constexpr = 0,
):
    tl.static_assert(
        thread_mode == 0 or coop == 0,
        "GDA aggregate thread mode requires thread cooperative scope",
    )
    if thread_mode == 1:
        if signal_op == 0:
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
        elif signal_op == 1:
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
    elif coop == 0:
        if signal_op == 0:
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
        elif signal_op == 1:
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
    elif coop == 1:
        if signal_op == 0:
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
        elif signal_op == 1:
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
        if signal_op == 0:
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
        elif signal_op == 1:
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
    signal_op: tl.constexpr = 0,
    signal_id=0,
    signal_val=0,
    coop: tl.constexpr = 0,
    thread_mode: tl.constexpr = 0,
):
    tl.static_assert(
        thread_mode == 0 or coop == 0,
        "GDA aggregate thread mode requires thread cooperative scope",
    )
    if thread_mode == 1:
        if signal_op == 0:
            raw.gda_put_value_at_none(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        elif signal_op == 1:
            raw.gda_put_value_at_inc(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        else:
            raw.gda_put_value_at_add(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
    elif coop == 0:
        if signal_op == 0:
            raw.gda_put_value_it_none(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        elif signal_op == 1:
            raw.gda_put_value_it_inc(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        else:
            raw.gda_put_value_it_add(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
    elif coop == 1:
        if signal_op == 0:
            raw.gda_put_value_iw_none(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        elif signal_op == 1:
            raw.gda_put_value_iw_inc(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        else:
            raw.gda_put_value_iw_add(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
    else:
        if signal_op == 0:
            raw.gda_put_value_ib_none(
                dev_comm, ctx, peer, dst_win, dst_off, value, signal_id, signal_val
            )
        elif signal_op == 1:
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
    coop: tl.constexpr = 0,
    thread_mode: tl.constexpr = 0,
):
    tl.static_assert(
        thread_mode == 0 or coop == 0,
        "GDA aggregate thread mode requires thread cooperative scope",
    )
    if thread_mode == 1:
        raw.gda_get_at(
            dev_comm, ctx, peer, remote_win, remote_off, local_win, local_off, nbytes
        )
    elif coop == 0:
        raw.gda_get_it(
            dev_comm, ctx, peer, remote_win, remote_off, local_win, local_off, nbytes
        )
    elif coop == 1:
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
    signal_op: tl.constexpr = 1,
    signal_id=0,
    signal_val=0,
    coop: tl.constexpr = 0,
):
    tl.static_assert(signal_op != 0, "GDA signal requires INC or ADD")
    if coop == 0:
        if signal_op == 1:
            raw.gda_signal_thread_inc(
                dev_comm, ctx, peer, signal_id, signal_val
            )
        else:
            raw.gda_signal_thread_add(
                dev_comm, ctx, peer, signal_id, signal_val
            )
    elif coop == 1:
        if signal_op == 1:
            raw.gda_signal_warp_inc(dev_comm, ctx, peer, signal_id, signal_val)
        else:
            raw.gda_signal_warp_add(dev_comm, ctx, peer, signal_id, signal_val)
    else:
        if signal_op == 1:
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
    coop: tl.constexpr = 0,
):
    if coop == 0:
        raw.gda_wait_signal_thread(dev_comm, ctx, signal_id, least, bits)
    elif coop == 1:
        raw.gda_wait_signal_warp(dev_comm, ctx, signal_id, least, bits)
    else:
        raw.gda_wait_signal_block(dev_comm, ctx, signal_id, least, bits)


@triton.jit
def _gda_flush(
    dev_comm,
    ctx: tl.constexpr = 0,
    coop: tl.constexpr = 1,
):
    if coop == 2:
        raw.gda_flush_block(dev_comm, ctx)
    else:
        raw.gda_flush_warp(dev_comm, ctx)


@triton.jit
def _gda_flush_peer(
    dev_comm,
    peer,
    ctx: tl.constexpr = 0,
    coop: tl.constexpr = 1,
):
    if coop == 2:
        raw.gda_flush_peer_block(dev_comm, ctx, peer)
    else:
        raw.gda_flush_peer_warp(dev_comm, ctx, peer)


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
    coop: tl.constexpr = 0,
    signal: tl.constexpr = True,
    aggregate: tl.constexpr = False,
):
    flags = 1 if aggregate else 0
    if coop == 0:
        if signal:
            raw.sdma_put_thread(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
        else:
            raw.sdma_put_thread_ns(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
    elif coop == 1:
        if signal:
            raw.sdma_put_warp(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
        else:
            raw.sdma_put_warp_ns(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
    else:
        if signal:
            raw.sdma_put_block(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
        else:
            raw.sdma_put_block_ns(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
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
    coop: tl.constexpr = 0,
    signal: tl.constexpr = True,
    aggregate: tl.constexpr = False,
):
    flags = 1 if aggregate else 0
    if coop == 0:
        if signal:
            raw.sdma_get_thread(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
        else:
            raw.sdma_get_thread_ns(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
    elif coop == 1:
        if signal:
            raw.sdma_get_warp(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
        else:
            raw.sdma_get_warp_ns(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
    else:
        if signal:
            raw.sdma_get_block(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )
        else:
            raw.sdma_get_block_ns(
                dev_comm,
                peer,
                dst_win,
                dst_off,
                src_win,
                src_off,
                nbytes,
                qid,
                flags,
            )


@triton.jit
def _sdma_commit(
    dev_comm,
    peer,
    qid=0,
    coop: tl.constexpr = 0,
):
    if coop == 0:
        raw.sdma_commit_thread(dev_comm, peer, qid)
    elif coop == 1:
        raw.sdma_commit_warp(dev_comm, peer, qid)
    else:
        raw.sdma_commit_block(dev_comm, peer, qid)


@triton.jit
def _sdma_quiet(
    dev_comm,
    peer,
    coop: tl.constexpr = 0,
):
    if coop == 0:
        raw.sdma_quiet_thread(dev_comm, peer)
    elif coop == 1:
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


__all__ = [
    "DevComm",
    "Window",
    "Gda",
    "Sdma",
    "CoopScope",
    "SignalOp",
    "ThreadMode",
    "SdmaOptFlags",
]
