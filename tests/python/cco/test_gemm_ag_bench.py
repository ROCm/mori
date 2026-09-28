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
"""Numerical validation contracts for GEMM + all-gather benchmarks."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def bench(monkeypatch):
    pytest.importorskip("flydsl")
    path = Path(__file__).resolve().parents[3] / "benchmark/cco/flydsl/gemm_ag"
    monkeypatch.syspath_prepend(str(path))
    module = importlib.import_module("bench_gemm_ag")
    monkeypatch.setattr(
        module,
        "_operands",
        lambda rank, args: (torch.eye(2) * (rank + 1), torch.eye(2), None, None),
    )
    monkeypatch.setattr(module, "_reference", lambda a, b, sa, sb, args: a @ b.T)
    return module


def test_recv_validation_rejects_a_nonfinite_peer(bench):
    args = SimpleNamespace(m=2, tolerance=1e-5)
    recv = torch.cat([torch.eye(2), 2 * torch.eye(2)])
    assert bench._validate_recv(recv, args, 0, 2)[1]
    recv[2, 0] = float("nan")
    error, valid = bench._validate_recv(recv, args, 0, 2)
    assert not valid and error == float("inf")


def test_recv_validation_checks_changed_input_sign(bench):
    args = SimpleNamespace(m=2, tolerance=1e-5)
    recv = -torch.cat([torch.eye(2), 2 * torch.eye(2)])
    assert bench._validate_recv(recv, args, 0, 2, sign=-1)[1]
    assert not bench._validate_recv(recv, args, 0, 2, sign=1)[1]
