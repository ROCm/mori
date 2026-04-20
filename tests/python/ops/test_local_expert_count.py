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
import torch

import mori
from mori.ops.local_expert_count import launch_local_expert_count

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA/HIP device is required"
)


def _make_config(
    *,
    rank=1,
    world_size=4,
    num_experts_per_rank=3,
    num_experts_per_token=2,
    warp_num_per_block=2,
    block_num=4,
):
    return mori.cpp.EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
    )


def _expected_local_expert_count(config, indices, total_recv_token_num):
    expected = torch.zeros(config.num_experts_per_rank, dtype=torch.int32)
    expert_base = config.rank * config.num_experts_per_rank

    for token_indices in indices[:total_recv_token_num]:
        for expert_id in token_indices:
            local_expert = expert_id - expert_base
            if 0 <= local_expert < config.num_experts_per_rank:
                expected[local_expert] += 1

    return expected


def _launch_local_expert_count(
    config,
    indices,
    total_recv_token_num,
    *,
    output_fill=-1,
    block_num=-1,
    warp_per_block=-1,
):
    indices = torch.tensor(indices, dtype=torch.int32, device="cuda")
    total_recv_token_num = torch.tensor(
        [total_recv_token_num], dtype=torch.int32, device="cuda"
    )
    local_expert_count = torch.full(
        (config.num_experts_per_rank,),
        output_fill,
        dtype=torch.int32,
        device="cuda",
    )

    launch_local_expert_count(
        config,
        indices.data_ptr(),
        total_recv_token_num.data_ptr(),
        local_expert_count.data_ptr(),
        block_num=block_num,
        warp_per_block=warp_per_block,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    return local_expert_count.cpu()


@pytest.mark.parametrize(
    ("config", "indices", "total_recv_token_num", "expected"),
    [
        pytest.param(
            _make_config(
                rank=1, world_size=4, num_experts_per_rank=3, num_experts_per_token=2
            ),
            [
                [3, 0],
                [5, 4],
                [3, 5],
                [1, 5],
            ],
            4,
            [2, 1, 3],
            id="mixed-local-and-remote",
        ),
        pytest.param(
            _make_config(
                rank=2, world_size=4, num_experts_per_rank=4, num_experts_per_token=1
            ),
            [
                [8],
                [11],
                [9],
                [10],
            ],
            4,
            [1, 1, 1, 1],
            id="all-local-num-experts-per-token-1",
        ),
        pytest.param(
            _make_config(
                rank=1, world_size=4, num_experts_per_rank=3, num_experts_per_token=2
            ),
            [
                [0, 1],
                [2, 8],
                [7, 11],
            ],
            3,
            [0, 0, 0],
            id="all-remote",
        ),
        pytest.param(
            _make_config(
                rank=0, world_size=2, num_experts_per_rank=4, num_experts_per_token=4
            ),
            [
                [1, 1, 1, 1],
                [0, 2, 2, 5],
            ],
            2,
            [1, 4, 2, 0],
            id="duplicate-local-experts-within-token",
        ),
        pytest.param(
            _make_config(
                rank=1, world_size=4, num_experts_per_rank=2, num_experts_per_token=2
            ),
            [
                [2, 3],
                [3, 4],
                [2, 0],
            ],
            2,
            [1, 2],
            id="ignores-rows-beyond-total-recv-token-num",
        ),
        pytest.param(
            _make_config(
                rank=1, world_size=4, num_experts_per_rank=4, num_experts_per_token=8
            ),
            [
                [4, 5, 6, 7, 0, 1, 2, 3],
                [7, 6, 5, 4, 9, 10, 11, 12],
            ],
            2,
            [2, 2, 2, 2],
            id="num-experts-per-token-8",
        ),
    ],
)
def test_local_expert_count_matches_reference(
    config,
    indices,
    total_recv_token_num,
    expected,
):
    actual = _launch_local_expert_count(config, indices, total_recv_token_num)
    expected = torch.tensor(expected, dtype=torch.int32)
    reference = _expected_local_expert_count(config, indices, total_recv_token_num)

    assert torch.equal(reference, expected)
    assert torch.equal(actual, expected)


def test_local_expert_count_grid_stride_loop():
    config = _make_config(
        rank=1,
        world_size=4,
        num_experts_per_rank=4,
        num_experts_per_token=8,
        warp_num_per_block=4,
        block_num=8,
    )
    indices = []
    for _ in range(20):
        indices.append([4, 5, 6, 7, 0, 1, 2, 3])
    total_recv_token_num = len(indices)

    actual = _launch_local_expert_count(
        config,
        indices,
        total_recv_token_num,
        block_num=1,
        warp_per_block=1,
    )

    expected = _expected_local_expert_count(config, indices, total_recv_token_num)
    assert torch.equal(actual, expected)


def test_local_expert_count_resets_output_between_nonzero_launches():
    config = _make_config(
        rank=1,
        world_size=4,
        num_experts_per_rank=3,
        num_experts_per_token=2,
    )
    indices_first = torch.tensor(
        [
            [3, 4],
            [5, 0],
            [5, 8],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    total_first = torch.tensor([3], dtype=torch.int32, device="cuda")
    indices_second = torch.tensor(
        [
            [4, 4],
            [3, 3],
            [0, 1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    total_second = torch.tensor([3], dtype=torch.int32, device="cuda")
    local_expert_count = torch.full((3,), -1, dtype=torch.int32, device="cuda")

    launch_local_expert_count(
        config,
        indices_first.data_ptr(),
        total_first.data_ptr(),
        local_expert_count.data_ptr(),
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()
    assert torch.equal(
        local_expert_count.cpu(), torch.tensor([1, 1, 2], dtype=torch.int32)
    )

    local_expert_count.fill_(77)
    launch_local_expert_count(
        config,
        indices_second.data_ptr(),
        total_second.data_ptr(),
        local_expert_count.data_ptr(),
        stream=torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(
        local_expert_count.cpu(), torch.tensor([2, 2, 0], dtype=torch.int32)
    )


def test_local_expert_count_uses_config_launch_defaults():
    config = _make_config(
        rank=1,
        world_size=4,
        num_experts_per_rank=3,
        num_experts_per_token=2,
        warp_num_per_block=1,
        block_num=1,
    )
    indices = [
        [3, 4],
        [5, 5],
        [0, 2],
    ]

    actual = _launch_local_expert_count(config, indices, total_recv_token_num=3)
    expected = _expected_local_expert_count(config, indices, total_recv_token_num=3)

    assert torch.equal(actual, expected)


def test_local_expert_count_accepts_explicit_launch_overrides():
    config = _make_config(
        rank=1,
        world_size=4,
        num_experts_per_rank=3,
        num_experts_per_token=2,
        warp_num_per_block=1,
        block_num=1,
    )
    indices = [
        [3, 0],
        [5, 4],
        [3, 5],
        [1, 5],
    ]

    actual = _launch_local_expert_count(
        config,
        indices,
        total_recv_token_num=4,
        block_num=3,
        warp_per_block=2,
    )
    expected = _expected_local_expert_count(config, indices, total_recv_token_num=4)

    assert torch.equal(actual, expected)


def test_local_expert_count_rejects_invalid_launch_settings():
    config = _make_config(
        rank=1,
        world_size=4,
        num_experts_per_rank=3,
        num_experts_per_token=2,
        warp_num_per_block=0,
        block_num=0,
    )
    indices = torch.tensor([[3, 4]], dtype=torch.int32, device="cuda")
    total_recv_token_num = torch.tensor([1], dtype=torch.int32, device="cuda")
    local_expert_count = torch.full((3,), -1, dtype=torch.int32, device="cuda")

    with pytest.raises(RuntimeError, match="positive block and warp settings"):
        launch_local_expert_count(
            config,
            indices.data_ptr(),
            total_recv_token_num.data_ptr(),
            local_expert_count.data_ptr(),
            stream=torch.cuda.current_stream().cuda_stream,
        )
