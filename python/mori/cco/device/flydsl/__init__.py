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
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""FlyDSL device bindings for the cco GDA/LSA API.

Self-contained device-IR binding layer (independent of the shmem ``mori.ir``
machinery):

  * :mod:`._bindings`  — 1:1 FFI prototypes for ``libmori_cco_device.bc``.
  * :mod:`._internal`  — the ``_ffi`` factory + ``cco_struct`` decorator.
  * :mod:`.handles`    — OO handles ``DevComm`` / ``Window`` / ``Gda`` and the
                         ``CoopScope`` / ``SignalOp`` / ``ThreadMode`` enums.

Example (inside ``@flyc.kernel``)::

    import mori.cco.device.flydsl as cco

    gda = cco.DevComm(dev_comm).gda(0)
    gda.put(1, recv, 0, send, 0, nbytes,
            signal_op=cco.SignalOp.INC, signal_id=0, coop=cco.CoopScope.BLOCK)
"""

from mori.cco.device.bitcode import find_cco_bitcode, get_bitcode_path
from .handles import DevComm, Window, Gda, CoopScope, SignalOp, ThreadMode, Sdma
from . import _bindings

__all__ = [
    "DevComm",
    "Window",
    "Gda",
    "CoopScope",
    "SignalOp",
    "ThreadMode",
    "find_cco_bitcode",
    "get_bitcode_path",
    "_bindings",
    "Sdma",
]
