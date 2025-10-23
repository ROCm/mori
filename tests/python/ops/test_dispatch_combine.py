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
from tests.python.utils import TorchDistProcessManager, ExceptionWrapper
import torch
import torch.distributed as dist
from tqdm import tqdm


class EpDispatchCombineTestCase:
    def __init__(self, config, use_max_token_num=False):
        self.config = config
        self.device = torch.device("cuda", self.config.rank)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(123 + self.config.rank)
        self.check_dispatch = False

        if use_max_token_num:
            num_token = self.config.max_num_inp_token_per_rank
        else:
            num_token = torch.randint(
                1, self.config.max_num_inp_token_per_rank + 1, (1,)
            ).item()

        # Initialize all rank tokens
        self.num_token = num_token
        self.all_rank_num_token = [None] * self.config.world_size
        torch.distributed.all_gather_object(self.all_rank_num_token, num_token)

        # Initialize all rank indices, weights, input and scales
        (
            self.all_rank_input,
            self.all_rank_indices,
            self.all_rank_weights,
            self.all_rank_scales,
        ) = ([], [], [], [])
        scale_dtype = (
            torch.float8_e4m3fnuz if self.config.scale_type_size == 1 else torch.float32
        )
        for rank_num_token in self.all_rank_num_token:
            self.all_rank_input.append(
                torch.empty(
                    rank_num_token,
                    self.config.hidden_dim,
                    dtype=self.config.data_type,
                    device=self.device,
                )
            )
            self.all_rank_indices.append(
                torch.empty(
                    rank_num_token,
                    self.config.num_experts_per_token,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            self.all_rank_weights.append(
                torch.empty(
                    rank_num_token,
                    self.config.num_experts_per_token,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self.all_rank_scales.append(
                torch.empty(
                    rank_num_token,
                    self.config.scale_dim,
                    dtype=scale_dtype,
                    device=self.device,
                )
            )

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier()

    def quantize_input(self, input):
        # Quantize input with scale_dim
        input = input.view(input.size(0), self.config.scale_dim, -1)
        scales = input.abs().amax(dim=-1, keepdim=True).to(torch.float32)
        scales.clamp_(min=1e-12)
        input = (input / scales).to(torch.float8_e4m3fnuz).view(self.num_token, -1)
        return input, scales.squeeze(-1)

    def dequantize_input(self, input, scales, dtype):
        # Quantize input with scale_dim
        if scales.dtype == torch.float8_e4m3fnuz:
            scales = scales.to(torch.float32)
        reshaped_input = input.view(input.size(0), self.config.scale_dim, -1).to(
            torch.float32
        )
        dequant_input = (
            (reshaped_input * scales.unsqueeze(-1)).to(dtype).view(*input.shape)
        )
        return dequant_input

    def gen_test_data(self):
        # gen indices
        indices = torch.empty(
            self.num_token,
            self.config.num_experts_per_token,
            dtype=torch.int64,
            # device=self.device,
        )
        for i in range(self.num_token):
            perm = torch.randperm(
                self.config.num_experts_per_rank * self.config.world_size,
                generator=self.rng,
                device=self.device,
            )
            indices[i] = perm[: self.config.num_experts_per_token]
        indices = indices.to(torch.int32).to(self.device)

        # gen weights
        weights = torch.rand(
            self.num_token,
            self.config.num_experts_per_token,
            dtype=torch.float32,
            generator=self.rng,
            device=self.device,
        )

        # gen input & output
        # some functions such as randn and cat are not implemented for fp8
        input = torch.randn(
            self.num_token,
            self.config.hidden_dim,
            dtype=torch.float32,
            generator=self.rng,
            device=self.device,
        )

        if self.config.data_type == torch.float8_e4m3fnuz and self.config.scale_dim > 0:
            input, scales = self.quantize_input(input)
            if self.config.scale_type_size == 1:
                scales = scales.to(torch.float8_e4m3fnuz)
        else:
            input = input.to(self.config.data_type)
            scales = None

        return (
            input,
            indices,
            weights,
            scales,
        )

    def check_dispatch_result(
        self,
        test_data,
        dispatch_output,
        dispatch_weights,
        dispatch_scales,
        dispatch_indices,
        dispatch_recv_num_token,
        src_token_pos,
    ):
        (
            input,
            indices,
            weights,
            scales,
        ) = test_data

        dist.all_gather(self.all_rank_input, input)
        dist.all_gather(self.all_rank_indices, indices)
        if dispatch_weights is not None:
            dist.all_gather(self.all_rank_weights, weights)
        if dispatch_scales is not None:
            dist.all_gather(self.all_rank_scales, scales)

        for i, pos in enumerate(src_token_pos):
            src_rank = int(pos) // self.config.max_num_inp_token_per_rank
            src_id = int(pos) % self.config.max_num_inp_token_per_rank
            assert torch.equal(
                self.all_rank_input[src_rank][src_id], dispatch_output[i]
            )
            if dispatch_weights is not None:
                assert torch.equal(
                    self.all_rank_weights[src_rank][src_id], dispatch_weights[i]
                )
            if dispatch_scales is not None:
                assert torch.equal(
                    self.all_rank_scales[src_rank][src_id], dispatch_scales[i]
                )
            assert torch.equal(
                self.all_rank_indices[src_rank][src_id], dispatch_indices[i]
            )
        assert len(torch.unique(src_token_pos)) == len(src_token_pos)
        assert len(src_token_pos) == dispatch_recv_num_token[0]

    def check_combine_result(
        self, test_data, combine_output, combine_output_weight, round
    ):
        (
            input,
            indices,
            weights,
            scales,
        ) = test_data

        def _get_expected(input, indices, weights, scales, combine_weights):
            if input.dtype == torch.float8_e4m3fnuz:
                assert scales is not None
                input = self.dequantize_input(input, scales, dtype=torch.float32)

            expected_output = input * torch.sum(weights, dim=1, keepdim=True)
            expected_output = expected_output.to(torch.bfloat16)
            pes = indices // self.config.num_experts_per_rank
            unique_pes = torch.tensor(
                [torch.unique(x).numel() for x in pes],
                device=self.device,
                dtype=indices.dtype,
            )
            expected_weight = weights * unique_pes.unsqueeze(1)
            return expected_output, expected_weight

        expected_output, expected_weight = _get_expected(
            input, indices, weights, scales, combine_output_weight
        )

        combine_output = combine_output
        combine_output_weight = combine_output_weight

        result_match = torch.allclose(
            combine_output, expected_output, atol=1e-2, rtol=1e-2
        )
        weight_match = (
            torch.allclose(combine_output_weight, expected_weight, atol=1e-5, rtol=1e-5)
            if combine_output_weight is not None
            else True
        )

        if result_match and weight_match:
            return

        for i in range(self.num_token):
            result_match = torch.allclose(
                combine_output[i], expected_output[i], atol=1e-2, rtol=1e-2
            )
            if not result_match:
                error_msg = (
                    f"{round}-th Combine result mismatch for token {i}:\n"
                    f"  indices[{i}]: {indices[i].cpu().tolist()}\n"
                    f"  got: {combine_output[i]}\n"
                    f"  expected : {expected_output[i]}\n"
                )
                raise ValueError(error_msg)

            if combine_output_weight is not None:
                weight_match = torch.allclose(
                    combine_output_weight[i], expected_weight[i], atol=1e-5, rtol=1e-5
                )
                if not weight_match:
                    error_msg = (
                        f"{round}-th Combine weight mismatch for token {i}:\n"
                        f"  indices[{i}]: {indices[i].cpu().tolist()}\n"
                        f"  got_weight: {combine_output_weight[i]}\n"
                        f"  expected_weight: {expected_weight[i]}\n"
                    )
                    raise ValueError(error_msg)

    def run_mori(self, test_dataset, op):
        outputs = []
        if self.config.rank == 0:
            total_num_token = sum(self.all_rank_num_token)
            test_dataset = tqdm(
                test_dataset,
                desc="Running MORI dispatch/combine (#tokens={})".format(
                    total_num_token
                ),
            )
        for test_data in test_dataset:
            (
                input,
                indices,
                weights,
                scales,
            ) = test_data

            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = op.dispatch(
                input,
                weights,
                scales,
                indices,
            )

            if self.check_dispatch:
                dispatch_result = (
                    dispatch_output.clone(),
                    dispatch_weights.clone(),
                    dispatch_scales.clone() if dispatch_scales is not None else None,
                    dispatch_indices.clone(),
                    dispatch_recv_num_token.clone(),
                    op.src_token_pos.clone(),
                )
            else:
                dispatch_result = None

            num_recv_token = dispatch_recv_num_token.item()
            if num_recv_token == 0:
                weighted_output = torch.empty_like(dispatch_output)
            else:
                dispatch_output = dispatch_output[:num_recv_token]
                dispatch_weights = dispatch_weights[:num_recv_token]
                dispatch_indices = dispatch_indices[:num_recv_token]
                if dispatch_scales is not None:
                    dispatch_scales = dispatch_scales[:num_recv_token]

                mask = (
                    self.config.num_experts_per_rank * self.config.rank
                    <= dispatch_indices
                ) & (
                    dispatch_indices
                    < self.config.num_experts_per_rank * (self.config.rank + 1)
                )
                mask = mask.to(self.device)
                if dispatch_output.dtype == torch.float8_e4m3fnuz:
                    dispatch_output = self.dequantize_input(
                        dispatch_output, dispatch_scales, dtype=torch.float32
                    )
                masked_weights = (mask * dispatch_weights).sum(dim=-1)
                weighted_output = dispatch_output * masked_weights.unsqueeze(1)

            weighted_output = weighted_output.to(torch.float32)
            combine_output, combine_output_weights = op.combine(
                weighted_output, dispatch_weights, dispatch_indices
            )

            combine_result = (
                combine_output[: self.num_token].clone().to(torch.bfloat16),
                combine_output_weights[: self.num_token].clone(),
            )
            outputs.append((dispatch_result, combine_result))

        return outputs

    def run_test(self, op, test_dataset):
        # Run mori dispathc/combine for all test data
        mori_results = self.run_mori(test_dataset, op)

        # Check mori results
        test_data_and_results = zip(test_dataset, mori_results)
        if self.config.rank == 0:
            test_data_and_results = tqdm(test_data_and_results, desc="Checking Result")

        for i, (test_data, mori_result) in enumerate(test_data_and_results):
            self.check_result(test_data, mori_result, i)

    def check_result(self, test_data, mori_result, round):
        dispatch_result, combine_result = mori_result
        # Check dispatch output
        if dispatch_result is not None:
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
                src_token_pos,
            ) = dispatch_result
            self.check_dispatch_result(
                test_data,
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
                src_token_pos,
            )
        # Check combine output
        combine_output, combine_weights = combine_result
        self.check_combine_result(test_data, combine_output, combine_weights, round)


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
    num_reps,
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
    test_data = [test_case.gen_test_data() for _ in range(num_reps)]
    test_case.run_test(op, test_data)


# TODO: create a sub process group so that we can test worlds size < 8
@pytest.mark.parametrize("world_size", (8,))
@pytest.mark.parametrize("data_type", (torch.float8_e4m3fnuz, torch.bfloat16))
@pytest.mark.parametrize("hidden_dim", (7168, 4096))
@pytest.mark.parametrize("scale_dim", (0, 32))
@pytest.mark.parametrize("scale_type_size", (1, 4))
@pytest.mark.parametrize("max_num_inp_token_per_rank", (1, 128, 2048))
@pytest.mark.parametrize("num_experts_per_rank", (32,))
@pytest.mark.parametrize("num_experts_per_token", (8,))
@pytest.mark.parametrize("num_reps", (1,))
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
    num_reps,
):
    if (data_type == torch.float8_e4m3fnuz) != (scale_dim > 0):
        pytest.skip("skip fp8 with scale_dim == 0")

    # Drain result queue if any result remains in the queue.
    result_queue = torch_dist_process_manager.result_queue
    while not result_queue.empty():
        try:
            result_queue.get_nowait()
        except Exception:
            break

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
                    num_reps,
                ],
            )
        )

    for i in range(world_size):
        (
            rank,
            result,
        ) = torch_dist_process_manager.result_queue.get()

        if result is not None:
            assert isinstance(result, ExceptionWrapper)
            torch_dist_process_manager.on_error = True
            result.reraise()
