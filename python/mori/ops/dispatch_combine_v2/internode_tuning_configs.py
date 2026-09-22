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

"""Compatibility lookup for the historical MI308X internode table.

The active HIP backend uses hip_tuning_configs.resolve_schedule(). Keep this
older import path without maintaining a second copy of the fallback rules.
"""

from .hip_tuning_configs import _legacy_internode_buckets


def lookup(world_size, hidden_dim, topk, num_tokens, dtype="fp8"):
    """Return historical dispatch/combine B/R/W, or None for an untuned shape.

    The legacy table has only fp8-dispatch/bf16-combine measurements and has
    always used them as the fallback for every dtype.
    """
    table = _legacy_internode_buckets(world_size, hidden_dim, topk)
    if not table:
        return None
    row = next(
        (row for row in table if row[0] is None or num_tokens <= row[0]), table[-1]
    )
    return {"dispatch": row[1], "combine": row[2]}
