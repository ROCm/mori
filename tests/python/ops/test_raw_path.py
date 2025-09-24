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

from mori import cpp as mori_cpp


@pytest.mark.skipif(
    mori_cpp.with_torch(), reason="Built with libtorch; raw path test not applicable"
)
def test_raw_api_smoke():
    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch not available in Python: {e}")

    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device not available")

    # Minimal config: single rank/node, tiny dims
    hidden_dim = 8
    max_tokens = 4
    cfg = mori_cpp.EpDispatchCombineConfig(
        rank=0,
        world_size=1,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=0,
        max_num_inp_token_per_rank=max_tokens,
        num_experts_per_rank=1,
        num_experts_per_token=1,
        warp_num_per_block=1,
        block_num=1,
        use_external_inp_buf=True,
    )

    handle = mori_cpp.EpDispatchCombineHandle(cfg)

    # Registered input buffer pointer and shape
    buf_ptr = mori_cpp.get_registered_input_buffer_raw(handle)
    dim0, dim1 = mori_cpp.get_registered_input_buffer_shape(handle)
    assert dim0 >= max_tokens and dim1 == hidden_dim

    # Build a descriptor and export to DLPack, then to a torch tensor
    desc = mori_cpp.MoriTensorDesc()
    # Populate via attribute assignment
    desc.data = buf_ptr
    desc.dim0 = dim0
    desc.dim1 = dim1
    desc.dtype = mori_cpp.MoriScalarType.Float32

    cap = mori_cpp.export_to_dlpack(desc, device_id=0)
    t = torch.utils.dlpack.from_dlpack(cap)
    assert t.is_cuda and list(t.shape) == [dim0, dim1]

    # Make sure we can write a small slice
    t[:1, :1].fill_(1.0)

    # Token mapping raw accessors should be callable
    s_map = mori_cpp.get_dispatch_sender_token_idx_map_raw(handle)
    r_map = mori_cpp.get_dispatch_receiver_token_idx_map_raw(handle)
    s_cap = mori_cpp.export_to_dlpack(s_map, device_id=0)
    r_cap = mori_cpp.export_to_dlpack(r_map, device_id=0)
    s = torch.utils.dlpack.from_dlpack(s_cap)
    r = torch.utils.dlpack.from_dlpack(r_cap)
    assert s.is_cuda and r.is_cuda

    # Reset call should be available
    mori_cpp.launch_reset_raw(handle)
