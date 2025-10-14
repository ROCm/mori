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
import pytest
import mori
import os
from tests.python.utils import TorchDistProcessManager
import torch
import torch.distributed as dist


class EpDispatchCombineTestCase:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda", self.config.rank)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(123)

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier()

    def gen_test_data(self, use_max_token_num=False):
        if use_max_token_num:
            num_token = torch.tensor(
                [
                    self.config.max_num_inp_token_per_rank
                    for i in range(self.config.world_size)
                ]
            ).to(self.device)
        else:
            num_token = torch.randint(
                0,
                self.config.max_num_inp_token_per_rank + 1,
                [self.config.world_size],
                generator=self.rng,
                device=self.device,
            )

        # gen indices
        all_rank_indices = []
        for r in range(self.config.world_size):
            indices = torch.empty(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.int64,
                # device=self.device,
            )
            for i in range(num_token[r]):
                perm = torch.randperm(
                    self.config.num_experts_per_rank * self.config.world_size,
                    generator=self.rng,
                    device=self.device,
                )
                indices[i] = perm[: self.config.num_experts_per_token]
            all_rank_indices.append(indices.to(torch.int32).to(self.device))

        # gen weights
        all_rank_weights = [
            torch.rand(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            for r in range(self.config.world_size)
        ]

        # gen scales
        all_rank_scales = [
            torch.rand(
                num_token[r],
                self.config.scale_dim,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            for r in range(self.config.world_size)
        ]
        if self.config.scale_type_size == 1:
            all_rank_scales = [t.to(torch.float8_e4m3fnuz) for t in all_rank_scales]

        # gen input & output
        # some functions such as randn and cat are not implemented for fp8
        all_rank_input = []
        for r in range(self.config.world_size):
            all_rank_input.append(
                torch.randn(
                    num_token[r],
                    self.config.hidden_dim,
                    dtype=torch.float32,
                    generator=self.rng,
                    device=self.device,
                ).to(self.config.data_type)
            )

        return (
            num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        )

    def run_test_once(self, op, test_data):
        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = op.dispatch(
            all_rank_input[self.config.rank],
            all_rank_weights[self.config.rank],
            all_rank_scales[self.config.rank],
            all_rank_indices[self.config.rank],
        )

        recv_num_token = dispatch_recv_num_token.item()
        max_expert_idx = dispatch_indices[:recv_num_token].max().item()
        num_experts = self.config.num_experts_per_rank * self.config.world_size
        if max_expert_idx >= num_experts:
            print(f"Invalid expert id: {max_expert_idx}")
            assert False

        combine_output, combine_output_weight = op.combine(
            dispatch_output, dispatch_weights, dispatch_indices, call_reset=True
        )
        self.sync()


@pytest.fixture(scope="session")
def torch_dist_process_manager():
    os.environ["MORI_DISABLE_P2P"] = "1"
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to spawn")
    except RuntimeError:
        pass
    manager = TorchDistProcessManager()
    manager.start_workers(world_size=8)
    yield manager
    manager.shutdown()


def _test_dispatch_combine(
    rank,
    world_size,
    data_type,
    hidden_dim,
    scale_dim,
    scale_type_size,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
):
    config = mori.ops.EpDispatchCombineConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        max_token_type_size=2,
        block_num=16,
        warp_num_per_block=16,
        kernel_type=mori.ops.EpDispatchCombineKernelType.InterNode,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    test_case = EpDispatchCombineTestCase(config)
    test_data = test_case.gen_test_data(True)
    num_reps = 2048
    for idx in range(num_reps):
        test_case.run_test_once(op, test_data)
        if rank == 0:
            print(f"Passed {idx}/{num_reps}")


# TODO: create a sub process group so that we can test worlds size < 8
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", (torch.float8_e4m3fnuz,))
@pytest.mark.parametrize("hidden_dim", (7168,))
@pytest.mark.parametrize("scale_dim", (56,))
@pytest.mark.parametrize("scale_type_size", (4,))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (4096,))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
def test_dispatch_combine(
    torch_dist_process_manager,
    world_size,
    data_type,
    hidden_dim,
    scale_dim,
    scale_type_size,
    max_num_inp_token_per_rank,
    num_experts_per_rank,
    num_experts_per_token,
):
    for i in range(world_size):
        torch_dist_process_manager.task_queue.put(
            (
                _test_dispatch_combine,
                [
                    world_size,
                    data_type,
                    hidden_dim,
                    scale_dim,
                    scale_type_size,
                    max_num_inp_token_per_rank,
                    num_experts_per_rank,
                    num_experts_per_token,
                ],
            )
        )

    results = []
    for i in range(world_size):
        (
            rank,
            result,
        ) = torch_dist_process_manager.result_queue.get()
        results.append(result)

    for result in results:
        if result is not None:
            pytest.assume(False, result)
