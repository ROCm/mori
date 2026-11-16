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
"""The interface an implementation exposes to the gemm_ar_bf16 case.

``op_harness.py`` drives anything with this shape, so the existing operator (through
``existing.py``) and a candidate are timed and checked by the same code.

What it computes: every rank ``r`` holds ``a_r`` (``[M, K]`` bf16) and its weight ``b_r``
(``[N, K]`` bf16, row-major, as a checkpoint keeps it); every rank receives
``y = sum over r of a_r @ b_r.T`` as ``[M, N]`` bf16. That is a row-parallel linear layer
followed by an all-reduce, with the all-reduce fused into the GEMM however the
implementation chooses.

Required:

``window_bytes_for(world_size, *, m_max, n, k) -> int``  (static)
    Bytes of cco window the implementation needs; the harness sizes
    ``per_rank_vmm = 2 * window + 256 MiB``.
``__init__(comm, *, n, k, m_max)``
    ``comm`` is an open ``mori.cco.Communicator``. ``1 <= M <= m_max`` for every call.
``set_weight(b)``
    Called once, before any call and outside any timing: keep, convert or reorder the
    weight as needed.
``__call__(a) -> torch.Tensor``
    ``a`` is ``[M, K]`` bf16 on the current device, any ``M`` in ``[1, m_max]``. Returns
    ``[M, N]`` bf16. The result may alias memory the implementation owns and may be
    overwritten by the next call. The call must be capturable in a CUDA graph; inside
    one, the returned tensor is the graph's output and every replay rewrites it.
    Running on the current stream; no host synchronisation inside.
``close()``
    Release the window and everything else.
``fused_path_ran``
    Attribute, ``True`` when the last call took the fused path, ``False`` when it fell
    back to an unfused one, ``None`` when the implementation cannot tell.

Optional, used when present:

``self_test()``
    A collective self-check; called once after construction.
``debug_preset_counters(value)``
    Set every running counter, flag and epoch the implementation keeps on the device to
    ``value``, so the harness can run calls across a 2^32 wrap. Without it the wrap
    test is reported "not covered".
"""

REQUIRED = ("window_bytes_for", "set_weight", "__call__", "close")
OPTIONAL = ("self_test", "debug_preset_counters")


def missing(cls):
    """Names of required members ``cls`` lacks; empty when it conforms."""
    return [name for name in REQUIRED if not callable(getattr(cls, name, None))]
