# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
#
# Based on PR #173 by Chao Chen <cchen104@amd.com>
# Adapted for refactored pybind function names.
"""Shim parity test: verifies that the Torch pybind shim and the XLA FFI shim
expose the same set of logical MoE operations.

Run with:
    python -m pytest tests/python/test_shim_parity.py -v
"""
import ctypes
import pathlib
import unittest

REQUIRED_OPS = {
    "dispatch",
    "combine",
    "dispatch_recv",
    "combine_recv",
    "reset",
}

FFI_SYMBOLS = {
    "dispatch": "mori_ep_dispatch",
    "combine": "mori_ep_combine",
    "dispatch_recv": "mori_ep_dispatch_recv",
    "combine_recv": "mori_ep_combine_recv",
    "reset": "mori_ep_reset",
}

TORCH_FUNCTIONS = {
    "dispatch": "prepare_and_build_args",
    "combine": "prepare_and_build_args",
    "dispatch_recv": "prepare_and_build_args",
    "combine_recv": "prepare_and_build_args",
    "reset": "launch_reset",
}


def _find_so(name: str):
    """Search for a .so in common build output locations."""
    mori_root = pathlib.Path(__file__).resolve().parent.parent.parent
    candidates = [
        mori_root / "build" / f"src/ffi/{name}",
        mori_root / "build_ffi" / f"src/ffi/{name}",
        mori_root / "build" / f"src/pybind/{name}",
        mori_root / "python" / "mori" / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


class TestShimParity(unittest.TestCase):
    """Ensure both the XLA FFI shim and the Torch pybind shim cover the
    required set of logical MoE operations."""

    def test_ffi_symbols_present(self):
        """All required FFI C symbols exist in libmori_xla_ffi_ops.so."""
        lib_path = _find_so("libmori_xla_ffi_ops.so")
        if lib_path is None:
            self.skipTest("libmori_xla_ffi_ops.so not found (ENABLE_XLA_FFI=OFF?)")
        lib = ctypes.CDLL(lib_path)
        missing = []
        for op, sym in FFI_SYMBOLS.items():
            if not hasattr(lib, sym):
                missing.append(f"{op} -> {sym}")
        self.assertEqual(missing, [], f"Missing FFI symbols: {missing}")

    def test_ffi_handle_lifecycle(self):
        """Handle create/destroy C symbols exist in libmori_xla_ffi_ops.so."""
        lib_path = _find_so("libmori_xla_ffi_ops.so")
        if lib_path is None:
            self.skipTest("libmori_xla_ffi_ops.so not found")
        lib = ctypes.CDLL(lib_path)
        self.assertTrue(hasattr(lib, "mori_ffi_create_handle"))
        self.assertTrue(hasattr(lib, "mori_ffi_destroy_handle"))
        self.assertTrue(hasattr(lib, "mori_ffi_register_kernel_module"))

    def test_torch_functions_present(self):
        """All required Torch pybind functions exist."""
        try:
            from mori import cpp as mori_cpp
        except (ImportError, ModuleNotFoundError):
            self.skipTest("mori.cpp not available")
        self.assertTrue(hasattr(mori_cpp, "prepare_and_build_args"))
        self.assertTrue(hasattr(mori_cpp, "launch_reset"))
        self.assertTrue(hasattr(mori_cpp, "EpDispatchCombineHandle"))

    def test_required_ops_covered(self):
        """Both symbol maps cover every required op."""
        self.assertEqual(set(FFI_SYMBOLS.keys()), REQUIRED_OPS)
        self.assertEqual(set(TORCH_FUNCTIONS.keys()), REQUIRED_OPS)


if __name__ == "__main__":
    unittest.main()
