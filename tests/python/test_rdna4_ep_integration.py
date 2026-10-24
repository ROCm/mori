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
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Host dispatch wiring; requires the native library, but no GPU execution."""

from types import SimpleNamespace

import pytest
import torch

from mori.ops import dispatch_combine as ep
from mori.ops._rdna4_ep import Rdna4EpPolicy
from mori.tensor_utils import dtype_to_int


def config(**changes):
    values = dict(
        data_type=torch.bfloat16,
        rank=0,
        world_size=2,
        gpu_per_node=2,
        hidden_dim=4096,
        scale_dim=0,
        scale_type_size=1,
        max_token_type_size=2,
        max_num_inp_token_per_rank=256,
        num_experts_per_rank=128,
        num_experts_per_token=8,
        use_external_inp_buf=True,
        kernel_type=ep.EpDispatchCombineKernelType.IntraNode,
    )
    values.update(changes)
    return ep.EpDispatchCombineConfig(**values)


@pytest.mark.parametrize("world", [2, 4])
def test_phase_launch_defaults_and_explicit_overrides(world):
    op = object.__new__(ep.EpDispatchCombineOp)
    op.config = config(world_size=world, gpu_per_node=world)
    op._rdna4_ep = Rdna4EpPolicy(world, 4096, 256, "bf16")
    assert op._intranode_dispatch_kernel(
        "bf16", num_tokens=64
    ) == op._rdna4_ep.dispatch_kernel(64)
    for mode in ("AUTO", "MANUAL"):
        op.launch_config_mode = mode
        for dispatch in (True, False):
            fields = dict(
                num_tokens=64,
                is_intranode_dispatch=dispatch,
                is_intranode_combine=not dispatch,
            )
            bn, _, wpb = op._resolve_launch_params(-1, -1, -1, **fields)
            assert (bn, wpb) == op._rdna4_ep.launch(64, dispatch=dispatch)
            assert op._resolve_launch_params(12, 3, 8, **fields) == (12, 3, 8)


def test_disabled_policy_preserves_upstream_dispatch():
    op = object.__new__(ep.EpDispatchCombineOp)
    op._rdna4_ep = None
    assert op._intranode_dispatch_kernel("bf16") == "EpDispatchIntraNodeKernel_bf16"
    assert op._intranode_dispatch_kernel("bf16", stdmoe=True).endswith("bf16_stdmoe")


@pytest.mark.parametrize(
    "changes",
    [
        dict(world_size=8),
        dict(gpu_per_node=4),
        dict(num_experts_per_token=0),
        dict(num_experts_per_token=65),
        dict(scale_dim=1),
        dict(max_token_type_size=4),
        dict(use_external_inp_buf=False),
        dict(max_total_recv_tokens=256),
        dict(quant_type="fp8_direct_cast"),
        dict(kernel_type=ep.EpDispatchCombineKernelType.IntraNodeLL),
        dict(hidden_dim=2050),
        dict(data_type=torch.float32),
    ],
)
def test_invalid_config_fails_before_collectives(monkeypatch, changes):
    monkeypatch.setenv("MORI_RDNA4_EP", "1")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName="gfx1201"),
    )
    monkeypatch.setattr(ep, "_ep_comm", lambda: "shmem")
    monkeypatch.setattr(
        ep,
        "_ensure_jit_kernels",
        lambda _: pytest.fail("must validate before JIT/collectives"),
    )
    with pytest.raises(ValueError, match="MORI_RDNA4_EP"):
        ep.EpDispatchCombineOp(config(**changes))


@pytest.mark.parametrize(
    "arch,comm", [("gfx942", "shmem"), ("gfx1200", "shmem"), ("gfx1201", "cco")]
)
def test_arch_and_backend_fail_before_collectives(monkeypatch, arch, comm):
    monkeypatch.setenv("MORI_RDNA4_EP", "1")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(gcnArchName=arch)
    )
    monkeypatch.setattr(ep, "_ep_comm", lambda: comm)
    with pytest.raises(ValueError, match="MORI_RDNA4_EP"):
        ep.EpDispatchCombineOp(config())


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("topk", range(1, 65))
def test_supported_topk_reaches_jit(monkeypatch, world, topk):
    monkeypatch.setenv("MORI_RDNA4_EP", "1")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(gcnArchName="gfx1201"),
    )
    monkeypatch.setattr(ep, "_ep_comm", lambda: "shmem")
    monkeypatch.setattr(ep, "_detect_warp_size", lambda: 32)

    def stop_before_collectives(_):
        raise RuntimeError("supported configuration reached JIT")

    monkeypatch.setattr(ep, "_ensure_jit_kernels", stop_before_collectives)
    with pytest.raises(RuntimeError, match="supported configuration reached JIT"):
        ep.EpDispatchCombineOp(
            config(world_size=world, gpu_per_node=world, num_experts_per_token=topk)
        )


@pytest.mark.parametrize("phase", ["dispatch", "combine"])
def test_hidden_mismatch_rejected_before_native_calls(phase):
    op = object.__new__(ep.EpDispatchCombineOp)
    op.config = config()
    op._rdna4_ep = Rdna4EpPolicy(2, 4096, 256, "bf16")
    payload = torch.empty((1, 2056), dtype=torch.bfloat16)
    indices = torch.zeros((1, 8), dtype=torch.int32)
    with pytest.raises(ValueError, match="fixed config.hidden_dim"):
        if phase == "dispatch":
            op.dispatch(payload, None, None, indices)
        else:
            op.combine(payload, None, indices)


def test_fp16_tensor_view_mapping():
    assert dtype_to_int(torch.float16) == 6


@pytest.mark.parametrize("weights_enabled", [True, False])
def test_empty_rank_preserves_dispatch_weights_presence(monkeypatch, weights_enabled):
    op = object.__new__(ep.EpDispatchCombineOp)
    op.config = config(world_size=4, gpu_per_node=4)
    op._rdna4_ep = Rdna4EpPolicy(4, 4096, 256, "bf16")
    op._dispatch_out_ptrs = (0, 1234, 0, 0, 0)
    op._dispatch_rules = None
    op._handle = None
    monkeypatch.setattr(ep, "_current_stream", lambda: 0)

    def prepare(handle, **fields):
        assert fields["weight_ptr"] == (1234 if weights_enabled else 0)
        raise RuntimeError("captured preparation")

    monkeypatch.setattr(ep.mori_cpp, "prepare_inference_args", prepare)
    weights = torch.empty((0, 8)) if weights_enabled else None
    with pytest.raises(RuntimeError, match="captured preparation"):
        op.dispatch(
            torch.empty((0, 4096), dtype=torch.bfloat16),
            weights,
            None,
            torch.empty((0, 8), dtype=torch.int32),
        )


@pytest.mark.parametrize("phase", ["dispatch", "combine"])
def test_standard_moe_rejected_before_native_calls(phase):
    op = object.__new__(ep.EpDispatchCombineOp)
    op._rdna4_ep = Rdna4EpPolicy(4, 4096, 256, "bf16")
    with pytest.raises(ValueError, match="ordinary IntraNode"):
        if phase == "dispatch":
            op.dispatch_standard_moe(None, None, None, None)
        else:
            op.combine_standard_moe(None, None, None)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("topk", [1, 63, 64])
@pytest.mark.parametrize("weights_enabled", [True, False])
def test_empty_receive_preserves_combine_weights_presence(
    monkeypatch, world, topk, weights_enabled
):
    op = object.__new__(ep.EpDispatchCombineOp)
    op.config = config(world_size=world, gpu_per_node=world, num_experts_per_token=topk)
    op._rdna4_ep = Rdna4EpPolicy(world, 4096, 256, "bf16", topk)
    op._dispatch_out_ptrs = (0, 1234, 0, 0, 0)
    op._combine_rules = None
    op._qt_str = "none"
    op._handle = None
    op._get_cur_rank_num_token = lambda _: 17
    monkeypatch.setattr(ep, "_current_stream", lambda: 0)

    def prepare(handle, **fields):
        assert fields["num_tokens"] == 17
        assert fields["weight_ptr"] == (1234 if weights_enabled else 0)
        raise RuntimeError("captured preparation")

    monkeypatch.setattr(ep.mori_cpp, "prepare_inference_args", prepare)
    weights = torch.empty((0, topk)) if weights_enabled else None
    with pytest.raises(RuntimeError, match="captured preparation"):
        op.combine(
            torch.empty((0, 4096), dtype=torch.bfloat16),
            weights,
            torch.empty((17, topk), dtype=torch.int32),
        )
