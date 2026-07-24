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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# Standalone cross-node AllGather perf (no compute): per size, time one AllGather,
# report ms + algorithmic GB/s for RCCL vs one mori handle
# (--handle hostproxy=hp_sdma | device=ibgda_sdma). Bit-exact gate.
import argparse
import os
import sys
import time
import traceback
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import mori.shmem as shmem  # noqa: E402
from mori.ccl import HierAllGather  # noqa: E402

_LOGS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "logs")
)


def _build(mode, rank, ws, rpn, per, device):
    if mode == "device":
        return HierAllGather(
            my_pe=rank,
            npes=ws,
            ranks_per_node=rpn,
            input_buffer_size=per,
            output_buffer_size=per * ws,
            copy_output_to_user=True,
        )
    from mori.ccl.host_proxy_ag import HostProxyHierAllGather

    cap = max(per, int(os.environ.get("MORI_FSDP_HOSTPROXY_CAP_MB", "512")) * (1 << 20))
    return HostProxyHierAllGather(
        my_pe=rank,
        npes=ws,
        ranks_per_node=rpn,
        output_buffer_size=cap * ws,
        device=device,
    )


def _time_ms(fn, reps, warmup):
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def _worker(rank, ws, rpn, device, mode, sizes_mb, reps, warmup):
    maxb = max(sizes_mb) * 1024 * 1024
    per = maxb + 4096
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", str(per * ws * 3 + (1 << 28)))
    shmem.shmem_torch_process_group_init("default")
    h = _build(mode, rank, ws, rpn, per, device)
    main = torch.cuda.current_stream()
    tag = "hp_sdma" if mode == "hostproxy" else "ibgda_sdma"
    if rank == 0:
        print(
            f"[ag-perf] world={ws} rpn={rpn} num_nodes={h.num_nodes} mode={mode}({tag}) sizes_mb={sizes_mb}"
        )
    rows = []
    try:
        for mb in sizes_mb:
            numel = (mb * 1024 * 1024) // 4
            inp = torch.arange(numel, device=device, dtype=torch.float32) + rank * 131.0
            om = torch.empty(numel * ws, dtype=torch.float32, device=device)
            orf = torch.empty(numel * ws, dtype=torch.float32, device=device)
            dist.all_gather_into_tensor(orf, inp)
            assert h(inp, om, numel, main)
            main.synchronize()
            torch.cuda.synchronize()
            bx = bool(torch.equal(om, orf))
            assert bx, f"bitexact MISMATCH {mb}MB"
            r_ms = _time_ms(
                lambda: (dist.all_gather_into_tensor(orf, inp)), reps, warmup
            )
            m_ms = _time_ms(
                lambda: (h(inp, om, numel, main), main.synchronize()),  # noqa: F821
                reps,
                warmup,
            )
            out_gb = numel * ws * 4 / 1e9
            if rank == 0:
                print(
                    f"[ag-perf] {mb}MB | rccl={r_ms:.3f}ms ({out_gb/r_ms*1e3:.1f}GB/s) "
                    f"{tag}={m_ms:.3f}ms ({out_gb/m_ms*1e3:.1f}GB/s) | bitexact={bx}"
                )
                rows.append((mb, r_ms, m_ms))
            dist.barrier()
        if rank == 0:
            os.makedirs(_LOGS, exist_ok=True)
            with open(os.path.join(_LOGS, f"ag_perf_{tag}.csv"), "w") as f:
                f.write("size_mb,rccl_ms,%s_ms\n" % tag)
                for r in rows:
                    f.write("%d,%.4f,%.4f\n" % r)
            print(f"[ag-perf] wrote {_LOGS}/ag_perf_{tag}.csv")
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        del h
        dist.barrier()
        shmem.shmem_finalize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--handle", choices=["hostproxy", "device"], default="hostproxy")
    p.add_argument(
        "--sizes-mb", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512]
    )
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    lr = int(os.environ.get("LOCAL_RANK", rank))
    rpn = int(os.environ.get("LOCAL_WORLD_SIZE", ws))
    torch.cuda.set_device(lr)
    device = torch.device(f"cuda:{lr}")
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", rank=rank, world_size=ws, device_id=device
    )
    torch._C._distributed_c10d._register_process_group(
        "default", torch.distributed.group.WORLD
    )
    try:
        _worker(rank, ws, rpn, device, a.handle, a.sizes_mb, a.reps, a.warmup)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
