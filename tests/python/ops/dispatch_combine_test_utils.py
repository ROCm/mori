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
from tests.python.utils import TorchDistProcessManager, data_type_supported
import mori
import torch
import torch.distributed as dist

TORCH_FLOAT4_E2M1FN_X2 = getattr(torch, "float4_e2m1fn_x2", None)


def _is_fp4x2_dtype(dtype):
    return TORCH_FLOAT4_E2M1FN_X2 is not None and dtype is TORCH_FLOAT4_E2M1FN_X2


_FP4_E2M1_LUT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def unpack_fp4x2(fp4x2_tensor, dtype=torch.bfloat16):
    """Unpack float4_e2m1fn_x2 tensor [*, H] to float [*, H*2].

    Each fp4x2 element (1 byte) stores two FP4 E2M1 values:
    low nibble = first value, high nibble = second value.
    """
    raw = fp4x2_tensor.view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=dtype, device=raw.device)
    result = torch.stack([lut[low.long()], lut[high.long()]], dim=-1)
    return result.reshape(*raw.shape[:-1], raw.shape[-1] * 2)


def _all_data_types():
    """Return parametrize list of all supported data types with skipif marks."""
    types = [
        torch.bfloat16,
        pytest.param(
            torch.float8_e4m3fnuz,
            marks=pytest.mark.skipif(
                not data_type_supported(torch.float8_e4m3fnuz),
                reason="Skip float8_e4m3fnuz, it is not supported",
            ),
        ),
        pytest.param(
            torch.float8_e4m3fn,
            marks=pytest.mark.skipif(
                not data_type_supported(torch.float8_e4m3fn),
                reason="Skip float8_e4m3fn, it is not supported",
            ),
        ),
    ]
    if TORCH_FLOAT4_E2M1FN_X2 is not None:
        types.append(
            pytest.param(
                TORCH_FLOAT4_E2M1FN_X2,
                marks=pytest.mark.skipif(
                    not data_type_supported(TORCH_FLOAT4_E2M1FN_X2),
                    reason="Skip float4_e2m1fn_x2, it is not supported",
                ),
            )
        )
    return types


def start_torch_dist_process_manager(world_size=8, disable_p2p=False):
    if disable_p2p:
        torch.cuda.empty_cache()
        import os

        os.environ["MORI_DISABLE_P2P"] = "1"

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to spawn")
    except RuntimeError:
        pass

    manager = TorchDistProcessManager()
    manager.start_workers(world_size=world_size)
    return manager


def assert_worker_results(manager, world_size):
    results = []
    for _ in range(world_size):
        rank, result = manager.result_queue.get()
        results.append((rank, result))

    for _, result in sorted(results, key=lambda item: item[0]):
        if result is not None:
            pytest.assume(False, result)


class EpDispatchCombineTestCase:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda", self.config.rank)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(123)

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier()

    def gen_test_data(self, use_max_token_num=False, routing="random"):
        """Generate test data."""
        if use_max_token_num:
            num_token = torch.tensor(
                [
                    self.config.max_num_inp_token_per_rank
                    for _ in range(self.config.world_size)
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

        total_experts = self.config.num_experts_per_rank * self.config.world_size

        all_rank_indices = []
        for r in range(self.config.world_size):
            n = int(num_token[r])
            if routing == "round_robin":
                indices = torch.empty(
                    n, self.config.num_experts_per_token, dtype=torch.int64
                )
                for i in range(n):
                    base = (
                        r * self.config.max_num_inp_token_per_rank + i
                    ) * self.config.num_experts_per_token
                    for j in range(self.config.num_experts_per_token):
                        indices[i, j] = (base + j) % total_experts
            elif routing == "spread":
                # Sends exactly one expert to every rank (requires num_experts_per_token ==
                # world_size). After per-rank deduplication each rank receives every source
                # token exactly once, so total recv = max_num_inp_token_per_rank * world_size
                # (the true worst case).
                assert (
                    self.config.num_experts_per_token == self.config.world_size
                ), "spread routing requires num_experts_per_token == world_size"
                indices = torch.empty(
                    n, self.config.num_experts_per_token, dtype=torch.int64
                )
                for i in range(n):
                    for j in range(self.config.num_experts_per_token):
                        indices[i, j] = j * self.config.num_experts_per_rank
            elif routing == "all_to_one":
                indices = torch.zeros(
                    n, self.config.num_experts_per_token, dtype=torch.int64
                )
            else:
                indices = torch.empty(
                    n, self.config.num_experts_per_token, dtype=torch.int64
                )
                for i in range(n):
                    perm = torch.randperm(
                        total_experts,
                        generator=self.rng,
                        device=self.device,
                    )
                    indices[i] = perm[: self.config.num_experts_per_token]
            all_rank_indices.append(indices.to(torch.int32).to(self.device))

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

        all_rank_input = []
        for r in range(self.config.world_size):
            input_fp32 = torch.randn(
                num_token[r],
                self.config.hidden_dim,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            if _is_fp4x2_dtype(self.config.data_type):
                fp4_bytes = torch.randint(
                    0,
                    256,
                    (num_token[r], self.config.hidden_dim),
                    dtype=torch.uint8,
                    generator=self.rng,
                    device=self.device,
                )
                all_rank_input.append(fp4_bytes.view(torch.float4_e2m1fn_x2))
            else:
                all_rank_input.append(input_fp32.to(self.config.data_type))

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
            _,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        src_token_pos = op.get_dispatch_src_token_pos()

        for i, pos in enumerate(src_token_pos):
            src_rank, src_id = op.decode_send_flat_idx(pos)
            if _is_fp4x2_dtype(self.config.data_type):
                assert torch.equal(
                    all_rank_input[src_rank][src_id].view(torch.uint8),
                    dispatch_output[i].view(torch.uint8),
                )
            else:
                assert torch.equal(all_rank_input[src_rank][src_id], dispatch_output[i])
            if dispatch_weights is not None:
                assert torch.equal(
                    all_rank_weights[src_rank][src_id], dispatch_weights[i]
                )
            if dispatch_scales is not None:
                assert torch.equal(
                    all_rank_scales[src_rank][src_id], dispatch_scales[i]
                )
            assert torch.equal(all_rank_indices[src_rank][src_id], dispatch_indices[i])
        assert len(torch.unique(src_token_pos)) == len(src_token_pos)
        assert len(src_token_pos) == dispatch_recv_num_token[0]

    def check_combine_result(
        self,
        op,
        test_data,
        combine_output,
        combine_output_weight=None,
        combine_data_type=None,
    ):
        self.sync()
        all_rank_num_token = test_data[0]
        all_rank_indices = test_data[1]
        all_rank_input = test_data[2]
        all_rank_weights = test_data[3]

        if combine_data_type is None:
            combine_data_type = self.config.data_type

        if _is_fp4x2_dtype(combine_data_type):
            return

        for i in range(all_rank_num_token[self.config.rank]):
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in all_rank_indices[self.config.rank][i].cpu().tolist()
            ]
            unique_pes = len(set(pes))

            inp = all_rank_input[self.config.rank][i]
            if _is_fp4x2_dtype(self.config.data_type):
                inp_converted = unpack_fp4x2(
                    inp.unsqueeze(0), dtype=combine_data_type
                ).squeeze(0)
            else:
                inp_converted = inp.to(combine_data_type)

            got, expected = combine_output[i], (
                inp_converted.to(torch.float32) * unique_pes
            ).to(combine_data_type)

            atol, rtol = 1e-2, 1e-2
            if getattr(self.config, "quant_type", "none") == "fp8_direct_cast":
                atol, rtol = 1e-1, 1e-1
            result_match = torch.allclose(
                got.float(), expected.float(), atol=atol, rtol=rtol
            )
            if not result_match:
                print(f"Rank[{self.config.rank}] result mismatch for token {i}:")
                print(
                    f"Rank[{self.config.rank}]   indices[{i}]: {all_rank_indices[self.config.rank][i].cpu().tolist()}"
                )
                print(f"Rank[{self.config.rank}]   pes: {pes}")
                print(f"Rank[{self.config.rank}]   unique_pes: {unique_pes}")
                print(f"Rank[{self.config.rank}]   got: {got}")
                print(f"Rank[{self.config.rank}]   expected : {expected}")
                print(
                    f"Rank[{self.config.rank}]   input : {all_rank_input[self.config.rank][i].to(torch.float32)}"
                )
            assert result_match

            if combine_output_weight is not None:
                got_weight, expected_weight = (
                    combine_output_weight[i],
                    all_rank_weights[self.config.rank][i] * unique_pes,
                )
                weight_match = torch.allclose(
                    got_weight, expected_weight, atol=1e-5, rtol=1e-5
                )
                if not weight_match:
                    print(f"Rank[{self.config.rank}] Weight mismatch for token {i}:")
                    print(
                        f"Rank[{self.config.rank}]   indices[{i}]: {all_rank_indices[self.config.rank][i].cpu().tolist()}"
                    )
                    print(f"Rank[{self.config.rank}]   pes: {pes}")
                    print(f"Rank[{self.config.rank}]   unique_pes: {unique_pes}")
                    print(f"Rank[{self.config.rank}]   got_weight: {got_weight}")
                    print(
                        f"Rank[{self.config.rank}]   expected_weight (weights[{i}] * {unique_pes}): {expected_weight}"
                    )
                assert weight_match

    def run_test_once(self, op, test_data, check_results=True):
        (
            _,
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
        if check_results:
            self.check_dispatch_result(
                op,
                test_data,
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            )

        total_recv_num_token = dispatch_recv_num_token[0].item()
        if not self.config.use_external_inp_buf:
            combine_input = op.get_registered_combine_input_buffer(
                self.config.data_type
            )
            combine_input[:total_recv_num_token, :].copy_(
                dispatch_output[:total_recv_num_token, :]
            )
        combine_output, combine_output_weight = op.combine(
            dispatch_output, dispatch_weights, dispatch_indices, call_reset=False
        )
        self.sync()
        if check_results:
            self.check_combine_result(
                op, test_data, combine_output, combine_output_weight
            )


def run_ep_dispatch_combine_test(
    config, test_case_cls, use_max_token_num=False, routing=None, check_results=True
):
    op = mori.ops.EpDispatchCombineOp(config)
    test_case = test_case_cls(config)
    gen_kwargs = {}
    if use_max_token_num:
        gen_kwargs["use_max_token_num"] = True
    if routing is not None:
        gen_kwargs["routing"] = routing
    test_data = test_case.gen_test_data(**gen_kwargs)
    test_case.run_test_once(op, test_data, check_results=check_results)
