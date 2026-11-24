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
"""The recv-reuse credit: a sender may write into a peer's recv only once that
peer has started the call, i.e. finished reading the previous one.

The end-of-call barrier (``kernels_sdma``) says every rank has *written* call
``e-1``. It does not say every rank has *read* it: the gathered result is the
recv itself, so a peer can pass the barrier, start call ``e``, and push into a
slot whose owner is still running whatever reads call ``e-1``. Nothing caught
this before the public op, because the benchmark replays identical operands and
a stale read of identical bytes is the right answer.

* **Post**, at the start of call ``e``: write ``e`` into slot ``rank`` of every
  peer's credit region. Being at the start of call ``e`` means this rank's
  stream has finished everything before it, the readers of ``e-1`` included.
* **Wait**, before the first byte goes to peer ``d``: spin until slot ``d`` of
  this rank's own credit region reads ``>= e``.

``e`` is the window's finished-call count (``flag_off``, written last by the
drain or gather) plus one, so it is the same device value on every launch path
and survives graph replay.

The contract: whatever reads the returned recv must be ordered before this
rank's next call on the device -- the same stream, or an event the next call
waits on.

Same protocol as ``gemm_a2a._credit``, duplicated rather than imported for the
reason that module gives. The helpers take the window's ``base``/``stride``
rather than the ``CachedWindow``: they are called inside runtime ``if``s, whose
captured state FlyDSL requires to be single MLIR values.
"""

import flydsl.expr as fx

from ..gemm_ar._compat import local_load_u32, signal_ptr, signal_store_u32


def lsa(base, stride, peer, offset):
    """``CachedWindow.lsa_ptr`` from its two fields."""
    return base + fx.Int64(peer) * stride + fx.Int64(offset)


def call_epoch(base, stride, rank: int, flag_off: int):
    """This call's number: the window's finished-call count plus one."""
    return fx.Int32(local_load_u32(signal_ptr(lsa(base, stride, rank, flag_off)))) + 1


def post_credit(base, stride, rank: int, peer, credit_off: int, epoch):
    """Tell ``peer`` this rank has started call ``epoch``."""
    signal_store_u32(signal_ptr(lsa(base, stride, peer, credit_off + rank * 4)), epoch)


def wait_credit(my_credits, peer, epoch, spin_until):
    """Block until ``peer`` has started call ``epoch``. ``my_credits`` is this
    rank's own credit region, taken out before the branch."""
    spin_until(signal_ptr(my_credits + fx.Int64(peer) * fx.Int64(4)), epoch)


__all__ = ["call_epoch", "lsa", "post_credit", "wait_credit"]
