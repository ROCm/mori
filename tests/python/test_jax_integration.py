# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""JAX integration tests for mori XLA FFI and shmem interop.

Requires:
    - ROCm GPU(s) available
    - jax[rocm] installed
    - libmori_xla_ffi_ops.so built (ENABLE_XLA_FFI=ON)

Run with:
    LD_LIBRARY_PATH=build_ffi/src/ops:build_ffi/src/shmem:build_ffi/src/application:build_ffi/src/io:build_ffi/src/ffi \
    MORI_XLA_FFI_LIB=build_ffi/src/ffi/libmori_xla_ffi_ops.so \
    PYTHONPATH=python \
    python -m pytest tests/python/test_jax_integration.py -v
"""
import ctypes
import os
import unittest


def _jax_available():
    try:
        import jax
        return len(jax.devices()) > 0
    except (ImportError, RuntimeError):
        return False


def _ffi_lib_available():
    try:
        from mori.jax._ffi_registry import _find_library
        _find_library()
        return True
    except (ImportError, FileNotFoundError):
        return False


@unittest.skipUnless(_jax_available(), "JAX with GPU not available")
class TestJaxArrayInterop(unittest.TestCase):
    """Test JAX array <-> raw GPU pointer conversion."""

    def test_jax_data_ptr_returns_nonzero(self):
        import jax.numpy as jnp
        from mori.jax.shmem import jax_data_ptr

        arr = jnp.ones((4, 8), dtype=jnp.float32)
        ptr = jax_data_ptr(arr)
        self.assertIsInstance(ptr, int)
        self.assertGreater(ptr, 0)

    def test_shmem_ptr_to_jax_roundtrip(self):
        import jax.numpy as jnp
        from mori.jax.shmem import jax_data_ptr, shmem_ptr_to_jax

        arr = jnp.ones((4, 8), dtype=jnp.float32)
        ptr = jax_data_ptr(arr)
        arr2 = shmem_ptr_to_jax(ptr, (4, 8), jnp.float32)
        self.assertEqual(arr2.shape, (4, 8))
        self.assertTrue(jnp.allclose(arr, arr2))

    def test_multi_dtype_roundtrip(self):
        import jax.numpy as jnp
        from mori.jax.shmem import jax_data_ptr, shmem_ptr_to_jax

        for dt in [jnp.float32, jnp.bfloat16, jnp.int32, jnp.float16]:
            with self.subTest(dtype=dt):
                arr = jnp.zeros((2, 3), dtype=dt)
                ptr = jax_data_ptr(arr)
                arr2 = shmem_ptr_to_jax(ptr, (2, 3), dt)
                self.assertEqual(arr2.shape, (2, 3))
                self.assertEqual(arr2.dtype, dt)

    def test_unsupported_dtype_raises(self):
        from mori.jax.shmem import shmem_ptr_to_jax
        with self.assertRaises(ValueError):
            shmem_ptr_to_jax(0xDEAD, (1,), "bad_dtype")


@unittest.skipUnless(_jax_available(), "JAX with GPU not available")
@unittest.skipUnless(_ffi_lib_available(), "libmori_xla_ffi_ops.so not found")
class TestFFIRegistration(unittest.TestCase):
    """Test FFI library loading and JAX target registration."""

    def test_load_library(self):
        from mori.jax._ffi_registry import _load_library
        lib = _load_library()
        self.assertIsNotNone(lib)

    def test_ffi_symbols_present(self):
        from mori.jax._ffi_registry import _load_library
        lib = _load_library()
        expected = [
            "mori_ep_dispatch", "mori_ep_combine",
            "mori_ep_dispatch_recv", "mori_ep_combine_recv",
            "mori_ep_reset",
            "mori_ffi_create_handle", "mori_ffi_destroy_handle",
            "mori_ffi_register_kernel_module",
        ]
        for sym in expected:
            self.assertTrue(hasattr(lib, sym), f"Missing symbol: {sym}")

    def test_register_ffi_targets(self):
        from mori.jax._ffi_registry import register_ffi_targets
        register_ffi_targets()


@unittest.skipUnless(_jax_available(), "JAX with GPU not available")
@unittest.skipUnless(_ffi_lib_available(), "libmori_xla_ffi_ops.so not found")
class TestFFILifecycle(unittest.TestCase):
    """Test JIT compilation, kernel module registration, and handle lifecycle."""

    def test_jit_compile_and_register(self):
        from mori.jit.core import compile_genco
        from mori.jax._ffi_registry import register_kernel_module

        hsaco = compile_genco("ep_intranode")
        self.assertTrue(os.path.isfile(hsaco))
        register_kernel_module(0, hsaco)

    def test_handle_create_destroy(self):
        from mori.jit.core import compile_genco
        from mori.jax._ffi_registry import _load_library, register_kernel_module
        from mori.jax.shmem import (
            _ensure_shmem_module, shmem_get_unique_id,
            shmem_init_attr, shmem_finalize,
        )

        _ensure_shmem_module()
        lib = _load_library()
        flag = lib.mori_ffi_shmem_init_flag_uniqueid()
        uid = shmem_get_unique_id()
        rc = shmem_init_attr(flag, 0, 1, uid)
        self.assertEqual(rc, 0)

        try:
            hsaco = compile_genco("ep_intranode")
            register_kernel_module(0, hsaco)

            handle_id = lib.mori_ffi_create_handle(
                0, 1, 4096, 0, 0, 4, 128, 1, 2, 1, 1, 0, 1, 0, 1,
            )
            self.assertGreater(handle_id, 0)
            lib.mori_ffi_destroy_handle(handle_id)
        finally:
            shmem_finalize()


if __name__ == "__main__":
    unittest.main()
