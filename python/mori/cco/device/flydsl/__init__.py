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
from .handles import DevComm, Window, Gda, CoopScope, SignalOp, ThreadMode
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
]
