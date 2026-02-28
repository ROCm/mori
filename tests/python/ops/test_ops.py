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
import torch
import pytest
import mori

from tests.python.utils import fp4_x2_to_fp32


@pytest.mark.parametrize("nelems", (32,))
@pytest.mark.parametrize("input_type", (torch.float,))
@pytest.mark.parametrize("output_type", (torch.float4_e2m1fn_x2,))
def test_cast(
    nelems,
    input_type,
    output_type,
):
    device = torch.device("cuda", 0)
    inp = torch.randn((nelems,), dtype=input_type, device=device)
    out = torch.empty((nelems // 2,), dtype=output_type, device=device)
    print(inp, out)
    mori.ops.cast(inp, out)
    torch.cuda.synchronize()
    print(inp, out)
    print(fp4_x2_to_fp32(out))
    print(out.view(dtype=torch.uint8))
