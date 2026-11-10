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
"""Simulated EP dispatch/combine (MORI_EP_SIM_ENABLE=1): runs without a GPU."""
import time

import pytest
import torch

from mori.ops.dispatch_combine import (
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
    EpDispatchCombineOp,
)

HIDDEN, TOPK, TOKENS = 1024, 8, 64
DISPATCH_US, COMBINE_US = 3000, 5000


@pytest.fixture(autouse=True)
def sim_env(monkeypatch):
    monkeypatch.setenv("MORI_EP_SIM_ENABLE", "1")
    monkeypatch.setenv("MORI_EP_SIM_DISPATCH_US", str(DISPATCH_US))
    monkeypatch.setenv("MORI_EP_SIM_COMBINE_US", str(COMBINE_US))


def _op(kernel=EpDispatchCombineKernelType.IntraNode):
    return EpDispatchCombineOp(
        EpDispatchCombineConfig(
            data_type=torch.bfloat16,
            rank=0,
            world_size=8,
            hidden_dim=HIDDEN,
            scale_dim=0,
            scale_type_size=0,
            max_token_type_size=2,
            max_num_inp_token_per_rank=TOKENS,
            num_experts_per_rank=32,
            num_experts_per_token=TOPK,
            kernel_type=kernel,
            gpu_per_node=8,
        )
    )


def _inputs(n):
    x = torch.zeros(n, HIDDEN, dtype=torch.bfloat16)
    w = torch.ones(n, TOPK, dtype=torch.float32)
    idx = torch.zeros(n, TOPK, dtype=torch.int32)
    return x, w, idx


def _timed(fn, *args, **kwargs):
    t0 = time.monotonic()
    out = fn(*args, **kwargs)
    return out, (time.monotonic() - t0) * 1e6


def test_ops_sleep_the_fixed_latency():
    op = _op()
    x, w, idx = _inputs(16)
    (out, out_w, out_s, out_i, total), us = _timed(op.dispatch, x, w, None, idx)
    assert DISPATCH_US <= us < 2 * DISPATCH_US
    n = op.max_num_tokens_to_recv()
    assert (
        out.shape == (n, HIDDEN)
        and out_w.shape == (n, TOPK)
        and out_i.shape == (n, TOPK)
    )
    assert out_s is None and int(total.item()) == 16

    (out, out_w), us = _timed(op.combine, out, out_w, idx)
    assert COMBINE_US <= us < 2 * COMBINE_US
    assert out.shape == (TOKENS, HIDDEN) and out_w.shape == (TOKENS, TOPK)


def test_asyncll_recv_waits_out_the_rest():
    op = _op(EpDispatchCombineKernelType.AsyncLL)
    x, w, idx = _inputs(16)
    t0 = time.monotonic()
    _, send_us = _timed(op.dispatch, x, w, None, idx)
    assert send_us < DISPATCH_US / 2
    op.dispatch_recv()
    assert (time.monotonic() - t0) * 1e6 >= DISPATCH_US

    op.combine(x, w, idx)
    time.sleep(2 * COMBINE_US / 1e6)  # work that hides the transfer
    _, recv_us = _timed(op.combine_recv)
    assert recv_us < 1000


def test_latency_must_be_set(monkeypatch):
    monkeypatch.delenv("MORI_EP_SIM_COMBINE_US")
    op = _op()
    x, w, idx = _inputs(4)
    with pytest.raises(RuntimeError, match="MORI_EP_SIM_COMBINE_US"):
        op.combine(x, w, idx)
