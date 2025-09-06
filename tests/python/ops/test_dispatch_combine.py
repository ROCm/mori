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

    def check_dispatch_result(
        self,
        op,
        test_data,
        dispatch_output,
        dispatch_weights,
        dispatch_scales,
        dispatch_indices,
        dispatch_recv_num_token,
    ):
        self.sync()
        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        src_token_pos = op.get_dispatch_src_token_pos()

        for i, pos in enumerate(src_token_pos):
            src_rank = int(pos) // self.config.max_num_inp_token_per_rank
            src_id = int(pos) % self.config.max_num_inp_token_per_rank
            assert torch.equal(all_rank_input[src_rank][src_id], dispatch_output[i])
            assert torch.equal(all_rank_weights[src_rank][src_id], dispatch_weights[i])
            if dispatch_scales is not None:
                assert torch.equal(
                    all_rank_scales[src_rank][src_id], dispatch_scales[i]
                )
            assert torch.equal(all_rank_indices[src_rank][src_id], dispatch_indices[i])
        assert len(torch.unique(src_token_pos)) == len(src_token_pos)
        assert len(src_token_pos) == dispatch_recv_num_token[0]

    def check_combine_result(
        self, op, test_data, combine_output, combine_output_weight=None
    ):
        self.sync()
        all_rank_num_token = test_data[0]
        all_rank_indices = test_data[1]
        all_rank_input = test_data[2]
        all_rank_weights = test_data[3]

        for i in range(all_rank_num_token[self.config.rank]):
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in all_rank_indices[self.config.rank][i].cpu().tolist()
            ]
            unique_pes = len(set(pes))

            got, expected = combine_output[i], (
                all_rank_input[self.config.rank][i].to(torch.float32) * unique_pes
            ).to(self.config.data_type)

            result_match = torch.allclose(
                got.float(), expected.float(), atol=1e-2, rtol=1e-2
            )
            if not result_match and self.config.rank == 0:
                print(f"Result mismatch for token {i}:")
                print(
                    f"  indices[{i}]: {all_rank_indices[self.config.rank][i].cpu().tolist()}"
                )
                print(f"  pes: {pes}")
                print(f"  unique_pes: {unique_pes}")
                print(f"  got: {got}")
                print(f"  expected : {expected}")

            if combine_output_weight is not None:
                got_weight, expected_weight = (
                    combine_output_weight[i],
                    all_rank_weights[self.config.rank][i] * unique_pes,
                )
                weight_match = torch.allclose(
                    got_weight, expected_weight, atol=1e-5, rtol=1e-5
                )
                if not weight_match and self.config.rank == 0:
                    print(f"Weight mismatch for token {i}:")
                    print(
                        f"  indices[{i}]: {all_rank_indices[self.config.rank][i].cpu().tolist()}"
                    )
                    print(f"  pes: {pes}")
                    print(f"  unique_pes: {unique_pes}")
                    print(f"  got_weight: {got_weight}")
                    print(
                        f"  expected_weight (weights[{i}] * {unique_pes}): {expected_weight}"
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
        self.sync()
        self.check_dispatch_result(
            op,
            test_data,
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        )

        combine_output, combine_output_weight = op.combine(
            dispatch_output, dispatch_weights, dispatch_indices, call_reset=False
        )
        self.sync()
        self.check_combine_result(op, test_data, combine_output, combine_output_weight)


@pytest.fixture(scope="session")
def torch_dist_process_manager():
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
        max_token_type_size=4,
        block_num=40,
        warp_num_per_block=8,
    )
    op = mori.ops.EpDispatchCombineOp(config)
    test_case = EpDispatchCombineTestCase(config)
    test_data = test_case.gen_test_data()
    test_case.run_test_once(op, test_data)


# TODO: create a sub process group so that we can test worlds size < 8
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", (torch.bfloat16,))
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize("scale_dim", (0, 32))
@pytest.mark.parametrize("scale_type_size", (1, 4))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 128))
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
