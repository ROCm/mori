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
"""The recv-reuse credit: a sender may overwrite a peer's recv only once that
peer has started the call, i.e. finished reading the previous one.

The end-of-call barrier (``kernels_sdma``) says every rank has *written* call
``e-1``. It does not say every rank has *read* it: a peer can pass the barrier,
start call ``e``, and push into a recv slot whose owner is still running the
attention that reads call ``e-1``. The credit closes that window.

* **Post**, at the start of call ``e``: write ``e`` into slot ``rank`` of each
  peer's credit region. Being at the start of call ``e`` means this rank's
  stream has finished everything before it, the readers of ``e-1`` included.
* **Wait**, before the first byte goes to peer ``d``: spin until slot ``d`` of
  this rank's own credit region reads ``>= e``.

``e`` is the window's call counter plus one: ``flag_off`` holds the last call
that finished (the drain or scatter writes it last), so it is the same device
value on every launch path and survives graph replay.

The contract this rests on: whatever reads the returned recv must be ordered
before this rank's next call on the device -- the same stream, or an event the
next call waits on. A reader on an unsynchronised stream is not covered.

Cost is a few remote stores at entry and, before each put, a load of a local
word that has almost always arrived: the post happens at kernel entry and the
first put waits for a whole wave of GEMM tiles.
"""

import flydsl.expr as fx

from mori.cco.device.flydsl.signals import (
    local_load_u32,
    signal_ptr,
    signal_store_u32,
)
from .kernels_lsa import _spin_until


def call_epoch(w, rank: int, flag_off: int):
    """This call's number: the window's finished-call count plus one."""
    return fx.Int32(local_load_u32(signal_ptr(fx.Int64(w.lsa_ptr(rank, flag_off))))) + 1


def post_credit(w, rank: int, peer, credit_off: int, epoch):
    """Tell ``peer`` this rank has started call ``epoch``."""
    slot = fx.Int64(w.lsa_ptr(peer, credit_off)) + fx.Int64(rank * 4)
    signal_store_u32(signal_ptr(slot), epoch)


def wait_credit(w, rank: int, peer, credit_off: int, epoch):
    """Block until ``peer`` has started call ``epoch``."""
    slot = fx.Int64(w.lsa_ptr(rank, credit_off)) + fx.Int64(peer) * fx.Int64(4)
    _spin_until(signal_ptr(slot), epoch)


__all__ = ["call_epoch", "post_credit", "wait_credit"]
