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
import struct

import pytest
import torch

from mori.ops.gemm_ar.gates import (
    GATE_ABI_VERSION,
    GATE_ROLE_PUBLISH,
    GATE_STRIDE,
    GATE_VIEW_HEADER_BYTES,
    GATE_VIEW_RECORD_BYTES,
    GateProducerSpec,
    GateTable,
    gate_view,
)


def test_producer_spec():
    spec = GateProducerSpec(mt0=96, mt1=128)
    assert spec.publishers(7168) == 75  # ceil(7168 / 96)
    assert GateProducerSpec(mt0=256, mt1=256).publishers(7168) == 28
    with pytest.raises(ValueError):
        GateProducerSpec(mt0=0, mt1=128)


def test_gate_table_targets_are_cumulative():
    t = GateTable(0x1000, 4)
    assert t.addr(3) == 0x1000 + 3 * GATE_STRIDE
    assert t.expect(0, 75) == 75
    assert t.expect(0, 75) == 150
    assert t.expect(2, 16) == 16
    assert t.expected.tolist() == [150, 0, 16, 0] and t.high() == 150
    t.expect_first(3, 1)
    assert t.expected.tolist() == [151, 1, 17, 0]
    t.reset()
    assert t.high() == 0
    with pytest.raises(ValueError):
        GateTable(0x1004, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gate_view_layout():
    heap = 0x7F0000000000
    gates = [heap + 0x100 + 64 * i for i in range(3)]
    gates.append(gates[0])  # several bands may share one gate
    view = gate_view(heap, gates, publishers=75, rank=5)
    raw = view.cpu().numpy().tobytes()
    assert len(raw) == GATE_VIEW_HEADER_BYTES + 4 * GATE_VIEW_RECORD_BYTES
    abi, size, rec, count, rank, reserved, epoch, base, tiles = struct.unpack_from(
        "<IIIIIIQQQ", raw, 0
    )
    assert (abi, size, rec, count, rank, reserved) == (
        GATE_ABI_VERSION,
        GATE_VIEW_HEADER_BYTES,
        GATE_VIEW_RECORD_BYTES,
        4,
        5,
        0,
    )
    assert base == heap and tiles == view.data_ptr() + GATE_VIEW_HEADER_BYTES
    for i, g in enumerate(gates):
        tid, ch, role, pubs, ready, arr, comp, err = struct.unpack_from(
            "<IIIIQQQQ", raw, GATE_VIEW_HEADER_BYTES + GATE_VIEW_RECORD_BYTES * i
        )
        assert (tid, ch, role, pubs) == (i, 0, GATE_ROLE_PUBLISH, 75)
        assert ready == g - heap and (arr, comp, err) == (0, 0, 0)
