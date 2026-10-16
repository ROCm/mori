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

"""Compatibility lookup backed by the HIP V2 internode JSON rules.

The active HIP backend uses hip_tuning_configs.resolve_schedule(). This older
entry point keeps its fp8-dispatch/bf16-combine defaults.
"""

import torch

from mori.ops import utils as _gpu
from mori.ops.tuning_config import CONFIG_STR_TO_DTYPE

from .hip_tuning_configs import lookup_internode


def lookup(
    world_size,
    hidden_dim,
    topk,
    num_tokens,
    dtype="fp8",
    *,
    kernel_family="v2",
    experts_per_rank=None,
):
    """Return dispatch/combine B/R/W from the selected family's JSON rules.

    ``dtype`` accepts a torch dtype, a shared config string, or ``"fp8"``
    (FNUZ on gfx942, OCP elsewhere). Combine remains BF16. Missing rules use
    the same None result as lookup_internode(), without borrowing another dtype.
    ``experts_per_rank`` must be supplied to match a measured expert/top-k shape;
    leaving it unknown returns None rather than selecting another model's rule.
    """
    if isinstance(dtype, str):
        if dtype == "fp8":
            dtype = (
                torch.float8_e4m3fnuz
                if _gpu.arch_name() == "gfx942"
                else torch.float8_e4m3fn
            )
        else:
            dtype = CONFIG_STR_TO_DTYPE.get(dtype)
            if dtype is None:
                return None
    return lookup_internode(
        world_size,
        hidden_dim,
        topk,
        num_tokens,
        dtype,
        torch.bfloat16,
        kernel_family=kernel_family,
        experts_per_rank=experts_per_rank,
    )
