# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""Shim parity test: verifies that the Torch pybind shim and the XLA FFI shim
expose the same set of logical MoE operations.

Run with:
    python -m pytest tests/python/test_shim_parity.py -v
"""
import ctypes
import os
import pathlib
import unittest

# Logical ops that both shims MUST cover
REQUIRED_OPS = {
    "dispatch",
    "combine",
    "dispatch_recv",
    "combine_recv",
    "reset",
}

# FFI handler C symbols exported from libmori_xla_ffi_ops.so
FFI_SYMBOLS = {
    "dispatch": "mori_ep_dispatch",
    "combine": "mori_ep_combine",
    "dispatch_recv": "mori_ep_dispatch_recv",
    "combine_recv": "mori_ep_combine_recv",
    "reset": "mori_ep_reset",
}

# Torch pybind Python function names exposed in libmori_pybinds
TORCH_FUNCTIONS = {
    "dispatch": "launch_dispatch",
    "combine": "launch_combine",
    "dispatch_recv": "launch_dispatch_recv",
    "combine_recv": "launch_combine_recv",
    "reset": "launch_reset",
}


def _find_so(name: str):
    """Search for a .so in common build output locations."""
    mori_root = pathlib.Path(__file__).resolve().parent.parent.parent
    candidates = [
        mori_root / "build" / f"src/ffi/{name}",
        mori_root / "build" / f"src/pybind/{name}",
        mori_root / "build_xla_ffi" / f"src/ffi/{name}",
        mori_root / "build_core" / f"src/ffi/{name}",
        mori_root / "build_torch" / f"src/pybind/{name}",
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

    def test_torch_functions_present(self):
        """All required Torch pybind functions exist in libmori_pybinds."""
        try:
            import torch  # noqa: F401
            from mori.cpp import libmori_pybinds as m  # type: ignore
        except (ImportError, ModuleNotFoundError):
            self.skipTest("torch or libmori_pybinds not available")
        missing = []
        for op, fn_name in TORCH_FUNCTIONS.items():
            if not hasattr(m, fn_name):
                missing.append(f"{op} -> {fn_name}")
        self.assertEqual(missing, [], f"Missing torch pybind functions: {missing}")

    def test_required_ops_covered(self):
        """Both symbol maps cover every required op."""
        self.assertEqual(set(FFI_SYMBOLS.keys()), REQUIRED_OPS)
        self.assertEqual(set(TORCH_FUNCTIONS.keys()), REQUIRED_OPS)


if __name__ == "__main__":
    unittest.main()
