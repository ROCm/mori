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
"""M0 of the stage1 SDMA offload: where aiter's stage1 spends its time today.

Per phase of one MegaMoEV2 call -- compact prepare (quant + plan), the fused stage1
(CU payload producers + GEMM1) and stage2+combine -- the ms averaged over ranks, with the
stage1 config that was picked. The stage1 SDMA offload can only win what the prepare
and the payload producers cost beyond the GEMM1 itself.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=8 baseline.py \
        --network glm52 --bs-list 1024,2048,4096,8192
"""

import argparse
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "11_flydsl_mega_moe_sdma")
)

import main as ex11  # noqa: E402  (sets the shmem env defaults, imports torch first)
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import aiter.ops.flydsl.kernels.mega_moe  # noqa: E402,F401
import flydsl.expr as fx  # noqa: E402


def _phases(moe, x, wts, ids, run_tokens):
    """One eager call split in three, with a cuda event between the phases."""
    stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
    config = moe._select_config(run_tokens)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    ev[0].record()
    moe._run_compact_prepare(ids, run_tokens, config.stage1, stream, quant_input=x)
    ev[1].record()
    moe._run_fused_stage1(
        moe._s1_quant_x[:run_tokens],
        wts,
        moe._s1_quant_scale[:run_tokens],
        ids,
        stream=stream,
        config=config.stage1,
        prepared=True,
    )
    ev[2].record()
    moe._run_stage2(run_tokens, None, True, config, None)
    ev[3].record()
    return ev, config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", choices=ex11.NETWORKS, default="glm52")
    p.add_argument("--bs-list", default="1024,2048,4096,8192")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    bss = [int(v) for v in args.bs_list.split(",")]

    rank, world, device = ex11._setup()
    net, moe = ex11._build_moe(args, max(bss), rank, world, device)
    rows = []
    for bs in bss:
        x, wts, ids = ex11._make_inputs(
            bs, net["model_dim"], net["experts"], net["topk"], rank, args.seed, device
        )
        for _ in range(3):
            _phases(moe, x, wts, ids, bs)
        ex11._sync()
        acc = torch.zeros(3, dtype=torch.float64)
        for _ in range(args.iters):
            ev, config = _phases(moe, x, wts, ids, bs)
            torch.cuda.synchronize()
            acc += torch.tensor(
                [ev[i].elapsed_time(ev[i + 1]) for i in range(3)], dtype=torch.float64
            )
            dist.barrier()
        acc /= args.iters
        dist.all_reduce(acc)
        acc /= world
        s1 = config.stage1
        rows.append((bs, *acc.tolist(), s1))
    if rank == 0:
        print(
            "\n| tokens/rank | prepare ms | stage1 ms | stage2+combine ms | total ms |"
        )
        print("|---:|---:|---:|---:|---:|")
        for bs, a, b, c, _ in rows:
            print(f"| {bs} | {a:.3f} | {b:.3f} | {c:.3f} | {a + b + c:.3f} |")
        print()
        for bs, *_, s1 in rows:
            print(f"bs={bs} stage1 config: {s1}")
    ex11._sync()


if __name__ == "__main__":
    main()
