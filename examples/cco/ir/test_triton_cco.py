#!/usr/bin/env python3
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
"""End-to-end Triton tests for the CCO LSA, SDMA, and GDA device APIs.

Examples::

    # LSA only
    torchrun --standalone --nproc_per_node=2 test_triton_cco.py --transport lsa

    # SDMA (MORI must be built with BUILD_CCO_SDMA=ON)
    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=2 \
        test_triton_cco.py --transport sdma

    # GDA on a single AINIC rail
    MORI_DEVICE_NIC=ionic MORI_DISABLE_TOPO=1 MORI_RDMA_DEVICES=rocep9s0 \
      torchrun --standalone --nproc_per_node=2 \
        test_triton_cco.py --transport gda

CCO does not need the shmem post-compile hook.  The device communicator and
window cross the Triton boundary explicitly as ``dc.ptr`` and ``win.handle``.
"""

import argparse
import ctypes
import os
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from mori.cco import (
    CCODevCommRequirements,
    Communicator,
    GDA_CONNECTION_FULL,
    GDA_CONNECTION_NONE,
    UniqueId,
)
from mori.ir.triton import cco
from mori.jit.hip_driver import _check, _get_hip_lib

NUM_ELEMS = 256
NBYTES = NUM_ELEMS * 4
BLOCK = 256
PER_RANK_VMM = 128 * 1024 * 1024

SEND_OFF = 0
LSA_RECV_OFF = SEND_OFF + NBYTES
SDMA_RECV_OFF = LSA_RECV_OFF + 2 * NBYTES
SDMA_GET_OFF = SDMA_RECV_OFF + 2 * NBYTES
GDA_RECV_OFF = SDMA_GET_OFF + NBYTES
GDA_GET_OFF = GDA_RECV_OFF + 2 * NBYTES
GDA_VALUE_OFF = GDA_GET_OFF + NBYTES
WINDOW_BYTES = GDA_VALUE_OFF + 8

_H2D, _D2H = 1, 2


def _copy(dst, src, nbytes, kind):
    _check(
        _get_hip_lib().hipMemcpy(
            dst, src, ctypes.c_size_t(nbytes), ctypes.c_int(kind)
        ),
        "hipMemcpy",
    )


def fill_u32(ptr, values):
    host = (ctypes.c_uint32 * len(values))(*values)
    _copy(ctypes.c_void_p(ptr), host, ctypes.sizeof(host), _H2D)


def zero(ptr, nbytes):
    host = (ctypes.c_uint8 * nbytes)()
    _copy(ctypes.c_void_p(ptr), host, nbytes, _H2D)


def read_u32(ptr, count):
    host = (ctypes.c_uint32 * count)()
    _copy(host, ctypes.c_void_p(ptr), ctypes.sizeof(host), _D2H)
    return list(host)


def read_u64(ptr):
    host = ctypes.c_uint64()
    _copy(ctypes.byref(host), ctypes.c_void_p(ptr), 8, _D2H)
    return host.value


@triton.jit
def query_kernel(dev_comm, out):
    tl.store(out + 0, cco.DevComm.rank(dev_comm))
    tl.store(out + 1, cco.DevComm.world_size(dev_comm))
    tl.store(out + 2, cco.DevComm.lsa_rank(dev_comm))
    tl.store(out + 3, cco.DevComm.lsa_size(dev_comm))


@triton.jit
def lsa_put_kernel(
    dev_comm,
    window,
    dst_lsa_rank: tl.constexpr,
    src_off: tl.constexpr,
    dst_off: tl.constexpr,
    n_elements: tl.constexpr,
    block: tl.constexpr,
):
    offs = tl.arange(0, block)
    mask = offs < n_elements
    my_lsa_rank = cco.DevComm.lsa_rank(dev_comm)
    src_addr = cco.Window.lsa_ptr(window, my_lsa_rank, src_off)
    dst_addr = cco.Window.lsa_ptr(window, dst_lsa_rank, dst_off)
    src = src_addr.to(tl.pointer_type(tl.uint32), bitcast=True)
    dst = dst_addr.to(tl.pointer_type(tl.uint32), bitcast=True)
    tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


@triton.jit
def sdma_put_variant_kernel(
    dev_comm,
    window,
    peer: tl.constexpr,
    dst_off: tl.constexpr,
    src_off: tl.constexpr,
    nbytes: tl.constexpr,
    scope: tl.constexpr,
    mode: tl.constexpr,
):
    # mode: 0=quiet, 1=quiet_queue, 2=no-signal + signaled drain,
    #       3=aggregate + commit + quiet_queue.
    if mode == 2:
        cco.Sdma.put(
            dev_comm,
            peer,
            window,
            dst_off,
            window,
            src_off,
            nbytes,
            0,
            coop=scope,
            signal=False,
        )
    cco.Sdma.put(
        dev_comm,
        peer,
        window,
        dst_off,
        window,
        src_off,
        nbytes,
        0,
        coop=scope,
        signal=True,
        aggregate=mode == 3,
    )
    if mode == 3:
        cco.Sdma.commit(dev_comm, peer, 0, coop=scope)
    if mode == 0 or mode == 2:
        cco.Sdma.quiet(dev_comm, peer, coop=scope)
    else:
        cco.Sdma.quiet_queue(dev_comm, peer, 0)


@triton.jit
def sdma_get_kernel(
    dev_comm,
    window,
    peer: tl.constexpr,
    dst_off: tl.constexpr,
    src_off: tl.constexpr,
    nbytes: tl.constexpr,
):
    cco.Sdma.get(
        dev_comm,
        peer,
        window,
        dst_off,
        window,
        src_off,
        nbytes,
        0,
        coop=cco.CoopScope.BLOCK,
    )
    cco.Sdma.quiet(dev_comm, peer, coop=cco.CoopScope.BLOCK)


@triton.jit
def gda_put_variant_kernel(
    dev_comm,
    window,
    peer: tl.constexpr,
    dst_off: tl.constexpr,
    src_off: tl.constexpr,
    nbytes: tl.constexpr,
    tc: tl.constexpr,
    signal_op: tl.constexpr,
    signal_id: tl.constexpr,
    do_wait: tl.constexpr,
):
    signal_val = 1 if signal_op == cco.SignalOp.ADD else 0
    if tc == 0:
        cco.Gda.put(
            dev_comm,
            peer,
            window,
            dst_off,
            window,
            src_off,
            nbytes,
            signal_op=signal_op,
            signal_id=signal_id,
            signal_val=signal_val,
            coop=cco.CoopScope.THREAD,
            thread_mode=cco.ThreadMode.INDEPENDENT,
        )
    elif tc == 1:
        cco.Gda.put(
            dev_comm,
            peer,
            window,
            dst_off,
            window,
            src_off,
            nbytes,
            signal_op=signal_op,
            signal_id=signal_id,
            signal_val=signal_val,
            coop=cco.CoopScope.WARP,
            thread_mode=cco.ThreadMode.INDEPENDENT,
        )
    elif tc == 2:
        cco.Gda.put(
            dev_comm,
            peer,
            window,
            dst_off,
            window,
            src_off,
            nbytes,
            signal_op=signal_op,
            signal_id=signal_id,
            signal_val=signal_val,
            coop=cco.CoopScope.BLOCK,
            thread_mode=cco.ThreadMode.INDEPENDENT,
        )
    else:
        cco.Gda.put(
            dev_comm,
            peer,
            window,
            dst_off,
            window,
            src_off,
            nbytes,
            signal_op=signal_op,
            signal_id=signal_id,
            signal_val=signal_val,
            coop=cco.CoopScope.THREAD,
            thread_mode=cco.ThreadMode.AGGREGATE,
        )
    if tc == 1:
        if signal_op == 0:
            cco.Gda.flush(dev_comm, coop=cco.CoopScope.WARP)
        else:
            cco.Gda.flush_peer(dev_comm, peer, coop=cco.CoopScope.WARP)
    else:
        if signal_op == 0:
            cco.Gda.flush(dev_comm, coop=cco.CoopScope.BLOCK)
        else:
            cco.Gda.flush_peer(dev_comm, peer, coop=cco.CoopScope.BLOCK)

    if signal_op != 0 and do_wait:
        if tc == 1:
            cco.Gda.wait_signal(
                dev_comm, signal_id, 1, coop=cco.CoopScope.WARP
            )
        elif tc == 2:
            cco.Gda.wait_signal(
                dev_comm, signal_id, 1, coop=cco.CoopScope.BLOCK
            )
        else:
            cco.Gda.wait_signal(
                dev_comm, signal_id, 1, coop=cco.CoopScope.THREAD
            )


@triton.jit
def gda_get_variant_kernel(
    dev_comm,
    window,
    peer: tl.constexpr,
    dst_off: tl.constexpr,
    src_off: tl.constexpr,
    nbytes: tl.constexpr,
    tc: tl.constexpr,
):
    if tc == 0:
        cco.Gda.get(
            dev_comm,
            peer,
            window,
            src_off,
            window,
            dst_off,
            nbytes,
            coop=cco.CoopScope.THREAD,
            thread_mode=cco.ThreadMode.INDEPENDENT,
        )
        cco.Gda.flush(dev_comm, coop=cco.CoopScope.THREAD)
    elif tc == 1:
        cco.Gda.get(
            dev_comm,
            peer,
            window,
            src_off,
            window,
            dst_off,
            nbytes,
            coop=cco.CoopScope.WARP,
            thread_mode=cco.ThreadMode.INDEPENDENT,
        )
        cco.Gda.flush_peer(dev_comm, peer, coop=cco.CoopScope.WARP)
    elif tc == 2:
        cco.Gda.get(
            dev_comm,
            peer,
            window,
            src_off,
            window,
            dst_off,
            nbytes,
            coop=cco.CoopScope.BLOCK,
            thread_mode=cco.ThreadMode.INDEPENDENT,
        )
        cco.Gda.flush(dev_comm, coop=cco.CoopScope.BLOCK)
    else:
        cco.Gda.get(
            dev_comm,
            peer,
            window,
            src_off,
            window,
            dst_off,
            nbytes,
            coop=cco.CoopScope.THREAD,
            thread_mode=cco.ThreadMode.AGGREGATE,
        )
        cco.Gda.flush_peer(dev_comm, peer, coop=cco.CoopScope.THREAD)


@triton.jit
def gda_put_value_variant_kernel(
    dev_comm,
    window,
    peer: tl.constexpr,
    dst_off: tl.constexpr,
    value: tl.constexpr,
    tc: tl.constexpr,
    signal_op: tl.constexpr,
    signal_id: tl.constexpr,
):
    signal_val = 1 if signal_op == cco.SignalOp.ADD else 0
    if tc == 0:
        cco.Gda.put_value(
            dev_comm,
            peer,
            window,
            dst_off,
            value,
            signal_op=signal_op,
            signal_id=signal_id,
            signal_val=signal_val,
            coop=cco.CoopScope.THREAD,
            thread_mode=cco.ThreadMode.INDEPENDENT,
        )
        cco.Gda.flush_peer(dev_comm, peer, coop=cco.CoopScope.THREAD)
    elif tc == 1:
        cco.Gda.put_value(
            dev_comm,
            peer,
            window,
            dst_off,
            value,
            signal_op=signal_op,
            signal_id=signal_id,
            signal_val=signal_val,
            coop=cco.CoopScope.WARP,
            thread_mode=cco.ThreadMode.INDEPENDENT,
        )
        cco.Gda.flush_peer(dev_comm, peer, coop=cco.CoopScope.WARP)
    elif tc == 2:
        cco.Gda.put_value(
            dev_comm,
            peer,
            window,
            dst_off,
            value,
            signal_op=signal_op,
            signal_id=signal_id,
            signal_val=signal_val,
            coop=cco.CoopScope.BLOCK,
            thread_mode=cco.ThreadMode.INDEPENDENT,
        )
        cco.Gda.flush_peer(dev_comm, peer, coop=cco.CoopScope.BLOCK)
    else:
        cco.Gda.put_value(
            dev_comm,
            peer,
            window,
            dst_off,
            value,
            signal_op=signal_op,
            signal_id=signal_id,
            signal_val=signal_val,
            coop=cco.CoopScope.THREAD,
            thread_mode=cco.ThreadMode.AGGREGATE,
        )
        cco.Gda.flush_peer(dev_comm, peer, coop=cco.CoopScope.THREAD)
    if signal_op != 0:
        cco.Gda.wait_signal(
            dev_comm, signal_id, 1, coop=cco.CoopScope.BLOCK
        )


@triton.jit
def gda_signal_variant_kernel(
    dev_comm,
    peer: tl.constexpr,
    scope: tl.constexpr,
    signal_op: tl.constexpr,
    signal_id: tl.constexpr,
    signal_value: tl.constexpr,
    out,
):
    cco.Gda.signal(
        dev_comm,
        peer,
        signal_op=signal_op,
        signal_id=signal_id,
        signal_val=signal_value,
        coop=scope,
    )
    cco.Gda.flush_peer(dev_comm, peer, coop=scope)
    cco.Gda.wait_signal(
        dev_comm, signal_id, signal_value, coop=scope
    )
    tl.store(out, cco.Gda.read_signal(dev_comm, signal_id))


@triton.jit
def gda_reset_signal_kernel(
    dev_comm,
    signal_id: tl.constexpr,
):
    cco.Gda.reset_signal(dev_comm, signal_id)


@triton.jit
def gda_read_signal_kernel(
    dev_comm,
    signal_id: tl.constexpr,
    out,
):
    tl.store(out, cco.Gda.read_signal(dev_comm, signal_id))


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return local_rank, rank, world_size, UniqueId.from_bytes(payload[0])


def expected_payload(source_rank):
    return [source_rank * 1000 + i for i in range(NUM_ELEMS)]


def check_payload(win, offset, source_rank, label):
    got = read_u32(win.local_ptr + offset, NUM_ELEMS)
    expected = expected_payload(source_rank)
    for index in (0, 1, NUM_ELEMS // 2, NUM_ELEMS - 1):
        assert got[index] == expected[index], (
            f"{label}: got[{index}]={got[index]}, expected={expected[index]}"
        )


def test_query(dc, rank, world_size, extern_libs):
    out = torch.empty(4, dtype=torch.int32, device="cuda")
    query_kernel[(1,)](dc.ptr, out, extern_libs=extern_libs)
    torch.cuda.synchronize()
    expected = torch.tensor(
        [rank, world_size, rank, world_size], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(out, expected)
    print(f"[rank {rank}] DevComm query PASS", flush=True)


def test_lsa(comm, dc, win, rank, world_size, extern_libs):
    peer = (rank + 1) % world_size
    source = (rank - 1 + world_size) % world_size
    dst_off = LSA_RECV_OFF + rank * NBYTES
    lsa_put_kernel[(1,)](
        dc.ptr,
        win.handle,
        dst_lsa_rank=peer,
        src_off=SEND_OFF,
        dst_off=dst_off,
        n_elements=NUM_ELEMS,
        block=BLOCK,
        extern_libs=extern_libs,
    )
    torch.cuda.synchronize()
    comm.barrier()
    check_payload(win, LSA_RECV_OFF + source * NBYTES, source, "LSA")
    print(f"[rank {rank}] LSA PASS", flush=True)


def test_sdma(comm, dc, win, rank, world_size, extern_libs):
    peer = (rank + 1) % world_size
    source = (rank - 1 + world_size) % world_size
    dst_off = SDMA_RECV_OFF + rank * NBYTES
    for scope, scope_name in enumerate(("thread", "warp", "block")):
        for mode, mode_name in enumerate(
            ("quiet", "quiet_queue", "no_signal", "aggregate")
        ):
            zero(win.local_ptr + SDMA_RECV_OFF, 2 * NBYTES)
            comm.barrier()
            sdma_put_variant_kernel[(1,)](
                dc.ptr,
                win.handle,
                peer=peer,
                dst_off=dst_off,
                src_off=SEND_OFF,
                nbytes=NBYTES,
                scope=scope,
                mode=mode,
                extern_libs=extern_libs,
                num_warps=1,
            )
            torch.cuda.synchronize()
            comm.barrier()
            check_payload(
                win,
                SDMA_RECV_OFF + source * NBYTES,
                source,
                f"SDMA put {scope_name}/{mode_name}",
            )

    sdma_get_kernel[(1,)](
        dc.ptr,
        win.handle,
        peer=peer,
        dst_off=SDMA_GET_OFF,
        src_off=SEND_OFF,
        nbytes=NBYTES,
        extern_libs=extern_libs,
        num_warps=1,
    )
    torch.cuda.synchronize()
    check_payload(win, SDMA_GET_OFF, peer, "SDMA get")
    print(f"[rank {rank}] SDMA PASS", flush=True)


def test_gda(comm, dc, win, rank, world_size, extern_libs):
    peer = (rank + 1) % world_size
    source = (rank - 1 + world_size) % world_size
    dst_off = GDA_RECV_OFF + rank * NBYTES
    compile_out = torch.zeros(1, dtype=torch.int64, device="cuda")

    # Compile/link every monomorphized data-path and signal variant.  Thread
    # independent scalar calls would intentionally execute once per hardware
    # lane, so only the cooperative block variants are launched for behavior.
    for tc in range(4):
        for signal_op in range(3):
            gda_put_variant_kernel.warmup(
                dc.ptr,
                win.handle,
                peer=peer,
                dst_off=dst_off,
                src_off=SEND_OFF,
                nbytes=NBYTES,
                tc=tc,
                signal_op=signal_op,
                signal_id=0,
                do_wait=True,
                grid=(1,),
                extern_libs=extern_libs,
                num_warps=1,
            )
            gda_put_value_variant_kernel.warmup(
                dc.ptr,
                win.handle,
                peer=peer,
                dst_off=GDA_VALUE_OFF,
                value=rank + 0xCC00,
                tc=tc,
                signal_op=signal_op,
                signal_id=0,
                grid=(1,),
                extern_libs=extern_libs,
                num_warps=1,
            )
        gda_get_variant_kernel.warmup(
            dc.ptr,
            win.handle,
            peer=peer,
            dst_off=GDA_GET_OFF,
            src_off=SEND_OFF,
            nbytes=NBYTES,
            tc=tc,
            grid=(1,),
            extern_libs=extern_libs,
            num_warps=1,
        )
    for scope in range(3):
        for signal_op in (1, 2):
            gda_signal_variant_kernel.warmup(
                dc.ptr,
                peer=peer,
                scope=scope,
                signal_op=signal_op,
                signal_id=1,
                signal_value=1 if signal_op == 1 else 3,
                out=compile_out,
                grid=(1,),
                extern_libs=extern_libs,
                num_warps=1,
            )

    zero(win.local_ptr + GDA_RECV_OFF, 2 * NBYTES)
    comm.barrier()
    gda_put_variant_kernel[(1,)](
        dc.ptr,
        win.handle,
        peer=peer,
        dst_off=dst_off,
        src_off=SEND_OFF,
        nbytes=NBYTES,
        tc=2,
        signal_op=1,
        signal_id=0,
        do_wait=False,
        extern_libs=extern_libs,
        num_warps=1,
    )
    torch.cuda.synchronize()
    comm.barrier()
    check_payload(win, GDA_RECV_OFF + source * NBYTES, source, "GDA put")

    signal_out = torch.zeros(1, dtype=torch.int64, device="cuda")
    gda_read_signal_kernel[(1,)](
        dc.ptr,
        signal_id=0,
        out=signal_out,
        extern_libs=extern_libs,
        num_warps=1,
    )
    torch.cuda.synchronize()
    assert signal_out.item() >= 1
    gda_reset_signal_kernel[(1,)](
        dc.ptr,
        signal_id=0,
        extern_libs=extern_libs,
        num_warps=1,
    )
    torch.cuda.synchronize()
    gda_read_signal_kernel[(1,)](
        dc.ptr,
        signal_id=0,
        out=signal_out,
        extern_libs=extern_libs,
        num_warps=1,
    )
    torch.cuda.synchronize()
    assert signal_out.item() == 0

    gda_put_variant_kernel[(1,)](
        dc.ptr,
        win.handle,
        peer=peer,
        dst_off=dst_off,
        src_off=SEND_OFF,
        nbytes=NBYTES,
        tc=2,
        signal_op=1,
        signal_id=1,
        do_wait=True,
        extern_libs=extern_libs,
        num_warps=1,
    )
    torch.cuda.synchronize()

    zero(win.local_ptr + GDA_GET_OFF, NBYTES)
    comm.barrier()
    gda_get_variant_kernel[(1,)](
        dc.ptr,
        win.handle,
        peer=peer,
        dst_off=GDA_GET_OFF,
        src_off=SEND_OFF,
        nbytes=NBYTES,
        tc=2,
        extern_libs=extern_libs,
        num_warps=1,
    )
    torch.cuda.synchronize()
    check_payload(win, GDA_GET_OFF, peer, "GDA get")

    zero(win.local_ptr + GDA_VALUE_OFF, 8)
    comm.barrier()
    value = rank + 0xCC00
    gda_put_value_variant_kernel[(1,)](
        dc.ptr,
        win.handle,
        peer=peer,
        dst_off=GDA_VALUE_OFF,
        value=value,
        tc=2,
        signal_op=0,
        signal_id=0,
        extern_libs=extern_libs,
        num_warps=1,
    )
    torch.cuda.synchronize()
    comm.barrier()
    assert read_u64(win.local_ptr + GDA_VALUE_OFF) == source + 0xCC00
    print(f"[rank {rank}] GDA PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=("lsa", "sdma", "gda", "all"),
        default="lsa",
    )
    args = parser.parse_args()

    _, rank, world_size, uid = setup_distributed()
    if world_size != 2:
        raise RuntimeError("The CCO Triton example requires exactly 2 ranks")
    if args.transport in ("sdma", "all") and os.environ.get(
        "MORI_ENABLE_SDMA", ""
    ).lower() not in ("1", "true", "on"):
        raise RuntimeError("SDMA requires MORI_ENABLE_SDMA=1")

    extern_libs = cco.get_extern_libs()
    gda_enabled = args.transport in ("gda", "all")
    sdma_enabled = args.transport in ("sdma", "all")

    try:
        with Communicator.init(
            world_size, rank, uid, per_rank_vmm=PER_RANK_VMM
        ) as comm:
            mem = comm.alloc_mem(WINDOW_BYTES)
            win = comm.register_window(mem.ptr, mem.size)
            zero(win.local_ptr, WINDOW_BYTES)
            fill_u32(win.local_ptr + SEND_OFF, expected_payload(rank))

            reqs = CCODevCommRequirements()
            reqs.gda_connection_type = (
                GDA_CONNECTION_FULL if gda_enabled else GDA_CONNECTION_NONE
            )
            reqs.gda_signal_count = 32 if gda_enabled else 0
            reqs.gda_counter_count = 0
            reqs.sdma_queue_count = 2 if sdma_enabled else 0
            dc = comm.create_dev_comm(reqs)

            comm.barrier()
            test_query(dc, rank, world_size, extern_libs)
            if args.transport in ("lsa", "all"):
                test_lsa(comm, dc, win, rank, world_size, extern_libs)
            if sdma_enabled:
                test_sdma(comm, dc, win, rank, world_size, extern_libs)
            if gda_enabled:
                test_gda(comm, dc, win, rank, world_size, extern_libs)
            comm.barrier()
            if rank == 0:
                print(
                    f"All CCO Triton {args.transport} tests PASSED on "
                    f"{world_size} ranks",
                    flush=True,
                )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    sys.exit(main())
