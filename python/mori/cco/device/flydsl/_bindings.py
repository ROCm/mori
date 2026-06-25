# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""1:1 FlyDSL FFI prototypes for ``src/cco/device/cco_device_wrapper.cpp``.

Each entry mirrors exactly one ``extern "C"`` symbol in
``libmori_cco_device.bc``. All arguments are scalars (FlyDSL FFI is scalar-only):
handles (``ccoDevComm*`` / ``ccoWindow_t``) are passed as ``uint64`` intptrs,
coop-scope and signal-op as ``int32`` enums (see :mod:`.enums`).

This is the codegen target / single source of truth for the binding ABI; the
OO wrappers in :mod:`.comm` and :mod:`.gda` call through these.
"""

from ._internal import _ffi

# ── LSA (intra-node P2P) ──
# cco_lsa_ptr(window, peerLsaRank, offset) -> peer's load/store-accessible VA.
# The LSA model: get this pointer, then load/store it directly in the kernel.
cco_lsa_ptr = _ffi("cco_lsa_ptr", ["uint64", "int32", "uint64"], "uint64", pure=True)

# ── ccoDevComm field accessors ──
cco_devcomm_rank = _ffi("cco_devcomm_rank", ["uint64"], "int32", pure=True)
cco_devcomm_world_size = _ffi("cco_devcomm_world_size", ["uint64"], "int32", pure=True)
cco_devcomm_lsa_rank = _ffi("cco_devcomm_lsa_rank", ["uint64"], "int32", pure=True)
cco_devcomm_lsa_size = _ffi("cco_devcomm_lsa_size", ["uint64"], "int32", pure=True)

# ── data path ──
# cco_gda_put(devComm, ctx, peer, dstWin, dstOff, srcWin, srcOff, bytes,
#             signalOp, signalId, signalVal, coopScope)
cco_gda_put = _ffi(
    "cco_gda_put",
    ["uint64", "int32", "int32", "uint64", "uint64", "uint64", "uint64", "uint64",
     "int32", "int32", "uint64", "int32"],
    "void",
)

# cco_gda_put_value(devComm, ctx, peer, dstWin, dstOff, value,
#                   signalOp, signalId, signalVal, coopScope)
cco_gda_put_value = _ffi(
    "cco_gda_put_value",
    ["uint64", "int32", "int32", "uint64", "uint64", "uint64",
     "int32", "int32", "uint64", "int32"],
    "void",
)

# cco_gda_get(devComm, ctx, peer, remoteWin, remoteOff, localWin, localOff, bytes, coopScope)
cco_gda_get = _ffi(
    "cco_gda_get",
    ["uint64", "int32", "int32", "uint64", "uint64", "uint64", "uint64", "uint64", "int32"],
    "void",
)

# cco_gda_signal(devComm, ctx, peer, signalOp, signalId, signalVal, coopScope)
cco_gda_signal = _ffi(
    "cco_gda_signal",
    ["uint64", "int32", "int32", "int32", "int32", "uint64", "int32"],
    "void",
)

# ── signal slot local ops ──
cco_gda_read_signal = _ffi(
    "cco_gda_read_signal", ["uint64", "int32", "int32", "int32"], "uint64"
)
cco_gda_reset_signal = _ffi("cco_gda_reset_signal", ["uint64", "int32", "int32"], "void")

# cco_gda_wait_signal(devComm, ctx, signalId, least, bits, coopScope)
cco_gda_wait_signal = _ffi(
    "cco_gda_wait_signal",
    ["uint64", "int32", "int32", "uint64", "int32", "int32"],
    "void",
)

# ── completion ──
cco_gda_flush = _ffi("cco_gda_flush", ["uint64", "int32", "int32"], "void")
cco_gda_flush_peer = _ffi("cco_gda_flush_peer", ["uint64", "int32", "int32", "int32"], "void")
