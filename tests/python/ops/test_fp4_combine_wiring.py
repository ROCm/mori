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
"""Lightweight (no-GPU) wiring guards for the blockwise-FP4 combine feature.

These protect the Python-side plumbing that selects the packed-FP4 combine kernels so a
refactor cannot silently drop the feature. The actual kernel numerics are covered by the
GPU test ``test_dispatch_combine_intranode.py`` (quant_type="fp4_blockwise").
"""

from mori.ops.dispatch_combine import _QUANT_TYPE_MAP, EpDispatchCombineQuantType


def test_fp4_blockwise_registered_in_quant_type_map():
    assert "fp4_blockwise" in _QUANT_TYPE_MAP
    # fp4_blockwise reuses the Fp8BlockwiseQuant config path (shared 1-byte-slot staging +
    # float scales); the packed-FP4 kernels are chosen at launch, not via a distinct enum.
    assert (
        _QUANT_TYPE_MAP["fp4_blockwise"] == EpDispatchCombineQuantType.Fp8BlockwiseQuant
    )


def test_fp8bwq_to_fp4bwq_kernel_name_mapping():
    # Mirrors the launch-time selection: kernel_name.replace("_fp8bwq", "_fp4bwq").
    # Every fp8bwq combine variant must have a matching fp4bwq name (registered in
    # ep_intranode.hip) so the mapping never yields an unregistered symbol.
    fp8_variants = [
        "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block128_vec8",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block256_vec8",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block128_vec8_top9",
        "EpCombineIntraNodeKernel_bf16_nop2p_fp8bwq_noweight_block256_vec8_top9",
    ]
    for name in fp8_variants:
        mapped = name.replace("_fp8bwq", "_fp4bwq")
        assert "_fp8bwq" not in mapped
        assert mapped.count("_fp4bwq") == 1
