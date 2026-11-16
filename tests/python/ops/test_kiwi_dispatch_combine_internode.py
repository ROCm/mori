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
"""Multi-node KIWI EP correctness check. Requires a BUILD_OPS_KIWI build.

Runs one torchrun process per rank, for example EP16 on two nodes:

    tools/run_internode_test.sh --rank <0|1> --master-addr <ip> --ifname <nic> \\
        --entry tests/python/ops/test_kiwi_dispatch_combine_internode.py \\
        --nproc-per-node 8 --cmd test --kernel-type hybrid --dtype bf16 \\
        --max-tokens 128

--kernel-type selects the KIWI remote-completion backend (rdma-only, hybrid or
auto). --num-qp is accepted for run_internode_test.sh and ignored.
"""
import argparse
import os

import torch
import torch.distributed as dist
from mori.ops import dispatch_combine_kiwi
from tests.python.ops.dispatch_combine_test_utils import (
    EpDispatchCombineTestCase,
    cross_dtype_hidden_dims,
)

_DATA_TYPES = {
    "bf16": torch.bfloat16,
    "fp8_e4m3_fnuz": torch.float8_e4m3fnuz,
    "fp8_e4m3": torch.float8_e4m3fn,
}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cmd", default="test", choices=["test"])
    parser.add_argument(
        "--kernel-type", default="hybrid", choices=["rdma-only", "hybrid", "auto"]
    )
    parser.add_argument("--dtype", default="bf16", choices=list(_DATA_TYPES))
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-qp", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=8)
    return parser.parse_args()


def _run_test(args, rank, world_size):
    data_type = _DATA_TYPES[args.dtype]
    # max_token_type_size=2 makes KIWI combine in bfloat16 for every dispatch dtype.
    combine_data_type = torch.bfloat16
    dispatch_hidden_dim, combine_hidden_dim, _ = cross_dtype_hidden_dims(
        args.hidden_dim, data_type, combine_data_type
    )
    config = dispatch_combine_kiwi.EpDispatchCombineKiwiConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=args.hidden_dim,
        scale_dim=56,
        scale_type_size=4,
        max_num_inp_token_per_rank=args.max_tokens,
        num_experts_per_rank=32,
        num_experts_per_token=args.topk,
        max_token_type_size=2,
    )
    op = dispatch_combine_kiwi.EpDispatchCombineKiwiOp(
        config, _rcomp_backend=args.kernel_type
    )
    test_case = EpDispatchCombineTestCase(
        config,
        combine_data_type=combine_data_type,
        combine_hidden_dim=combine_hidden_dim,
        dispatch_hidden_dim=dispatch_hidden_dim,
    )
    for idx in range(args.rounds):
        test_data = test_case.gen_test_data(use_max_token_num=idx == 0)
        test_case.run_test_once(op, test_data)
        if rank == 0:
            print(f"Passed {idx + 1}/{args.rounds}", flush=True)
    del op
    dist.barrier()


def main():
    args = _parse_args()
    if not dispatch_combine_kiwi.is_available():
        raise SystemExit("mori was built without BUILD_OPS_KIWI")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="cpu:gloo", rank=rank, world_size=world_size)
    try:
        if rank == 0:
            print(
                f"KIWI EP internode test: world_size={world_size} "
                f"rcomp_backend={args.kernel_type} dtype={args.dtype} "
                f"max_tokens={args.max_tokens} hidden_dim={args.hidden_dim} "
                f"topk={args.topk}",
                flush=True,
            )
        _run_test(args, rank, world_size)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
