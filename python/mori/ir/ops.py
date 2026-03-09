# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
# MIT License
"""
Mori shmem device function ABI metadata.

Single source of truth for all device function signatures exposed by
libmori_shmem_device.bc.  Framework-specific backends (Triton, FlyDSL, …)
generate their own wrappers from this table.

Each entry maps a short Python name to:
  symbol  – mangled C symbol in the bitcode
  args    – ordered list of parameter types (as strings understood by backends)
  ret     – return type string ("int32", "uint64", …)
  pure    – whether the function has no side effects (default False)

Type strings: "int32", "uint64", "int64", "uint32", "float32", "float64"
  Pointers are passed as "uint64" (intptr cast).
"""

MORI_DEVICE_FUNCTIONS = {
    # ==== Query ====
    "my_pe": {
        "symbol": "mori_shmem_my_pe",
        "args": [],
        "ret": "int32",
    },
    "n_pes": {
        "symbol": "mori_shmem_n_pes",
        "args": [],
        "ret": "int32",
        "pure": True,
    },

    # ==== Point-to-Point ====
    "ptr_p2p": {
        "symbol": "mori_shmem_ptr_p2p",
        "args": ["uint64", "int32", "int32"],
        "ret": "uint64",
    },
    "ptr": {
        "symbol": "mori_shmem_ptr",
        "args": ["uint64", "int32"],
        "ret": "uint64",
    },

    # ==== Synchronization ====
    "quiet_thread": {
        "symbol": "mori_shmem_quiet_thread",
        "args": [],
        "ret": "int32",
    },
    "quiet_thread_pe": {
        "symbol": "mori_shmem_quiet_thread_pe",
        "args": ["int32"],
        "ret": "int32",
    },
    "quiet_thread_pe_qp": {
        "symbol": "mori_shmem_quiet_thread_pe_qp",
        "args": ["int32", "int32"],
        "ret": "int32",
    },
    "fence_thread": {
        "symbol": "mori_shmem_fence_thread",
        "args": [],
        "ret": "int32",
    },
    "fence_thread_pe": {
        "symbol": "mori_shmem_fence_thread_pe",
        "args": ["int32"],
        "ret": "int32",
    },
    "fence_thread_pe_qp": {
        "symbol": "mori_shmem_fence_thread_pe_qp",
        "args": ["int32", "int32"],
        "ret": "int32",
    },
    "barrier_all_thread": {
        "symbol": "mori_shmem_barrier_all_thread",
        "args": [],
        "ret": "int32",
    },
    "barrier_all_block": {
        "symbol": "mori_shmem_barrier_all_block",
        "args": [],
        "ret": "int32",
    },

    # ==== PutNbi – Thread ====
    "putmem_nbi_thread": {
        "symbol": "mori_shmem_putmem_nbi_thread",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_uint32_nbi_thread": {
        "symbol": "mori_shmem_put_uint32_nbi_thread",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_uint64_nbi_thread": {
        "symbol": "mori_shmem_put_uint64_nbi_thread",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_float_nbi_thread": {
        "symbol": "mori_shmem_put_float_nbi_thread",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_double_nbi_thread": {
        "symbol": "mori_shmem_put_double_nbi_thread",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },

    # ==== PutNbi – Warp ====
    "putmem_nbi_warp": {
        "symbol": "mori_shmem_putmem_nbi_warp",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_uint32_nbi_warp": {
        "symbol": "mori_shmem_put_uint32_nbi_warp",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_uint64_nbi_warp": {
        "symbol": "mori_shmem_put_uint64_nbi_warp",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_float_nbi_warp": {
        "symbol": "mori_shmem_put_float_nbi_warp",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_double_nbi_warp": {
        "symbol": "mori_shmem_put_double_nbi_warp",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },

    # ==== PutNbi – Block ====
    "putmem_nbi_block": {
        "symbol": "mori_shmem_putmem_nbi_block",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_uint32_nbi_block": {
        "symbol": "mori_shmem_put_uint32_nbi_block",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_uint64_nbi_block": {
        "symbol": "mori_shmem_put_uint64_nbi_block",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_float_nbi_block": {
        "symbol": "mori_shmem_put_float_nbi_block",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "put_double_nbi_block": {
        "symbol": "mori_shmem_put_double_nbi_block",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },

    # ==== PutNbi with Signal ====
    "putmem_nbi_signal_thread": {
        "symbol": "mori_shmem_putmem_nbi_signal_thread",
        "args": ["uint64", "uint64", "uint64", "uint64", "uint64", "int32", "int32", "int32"],
        "ret": "int32",
    },
    "putmem_nbi_signal_warp": {
        "symbol": "mori_shmem_putmem_nbi_signal_warp",
        "args": ["uint64", "uint64", "uint64", "uint64", "uint64", "int32", "int32", "int32"],
        "ret": "int32",
    },
    "putmem_nbi_signal_block": {
        "symbol": "mori_shmem_putmem_nbi_signal_block",
        "args": ["uint64", "uint64", "uint64", "uint64", "uint64", "int32", "int32", "int32"],
        "ret": "int32",
    },

    # ==== Immediate Put ====
    "put_size_imm_nbi_thread": {
        "symbol": "mori_shmem_put_size_imm_nbi_thread",
        "args": ["uint64", "uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "int32_p": {
        "symbol": "mori_shmem_int32_p",
        "args": ["uint64", "int32", "int32", "int32"],
        "ret": "int32",
    },
    "int64_p": {
        "symbol": "mori_shmem_int64_p",
        "args": ["uint64", "int64", "int32", "int32"],
        "ret": "int32",
    },
    "uint32_p": {
        "symbol": "mori_shmem_uint32_p",
        "args": ["uint64", "uint32", "int32", "int32"],
        "ret": "int32",
    },
    "uint64_p": {
        "symbol": "mori_shmem_uint64_p",
        "args": ["uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "int_p": {
        "symbol": "mori_shmem_int_p",
        "args": ["uint64", "int32", "int32", "int32"],
        "ret": "int32",
    },
    "float_p": {
        "symbol": "mori_shmem_float_p",
        "args": ["uint64", "float32", "int32", "int32"],
        "ret": "int32",
    },
    "double_p": {
        "symbol": "mori_shmem_double_p",
        "args": ["uint64", "float64", "int32", "int32"],
        "ret": "int32",
    },

    # ==== Atomic NonFetch – Thread ====
    "atomic_size_nonfetch_thread": {
        "symbol": "mori_shmem_atomic_size_nonfetch_thread",
        "args": ["uint64", "uint64", "uint64", "int32", "int32", "int32"],
        "ret": "int32",
    },
    "atomic_uint32_nonfetch_thread": {
        "symbol": "mori_shmem_atomic_uint32_nonfetch_thread",
        "args": ["uint64", "uint32", "int32", "int32", "int32"],
        "ret": "int32",
    },
    "atomic_uint64_nonfetch_thread": {
        "symbol": "mori_shmem_atomic_uint64_nonfetch_thread",
        "args": ["uint64", "uint64", "int32", "int32", "int32"],
        "ret": "int32",
    },
    "atomic_int64_nonfetch_thread": {
        "symbol": "mori_shmem_atomic_int64_nonfetch_thread",
        "args": ["uint64", "int64", "int32", "int32", "int32"],
        "ret": "int32",
    },

    # ==== Atomic Fetch – Thread ====
    "atomic_uint32_fetch_thread": {
        "symbol": "mori_shmem_atomic_uint32_fetch_thread",
        "args": ["uint64", "uint32", "uint32", "int32", "int32", "int32"],
        "ret": "uint32",
    },
    "atomic_uint64_fetch_thread": {
        "symbol": "mori_shmem_atomic_uint64_fetch_thread",
        "args": ["uint64", "uint64", "uint64", "int32", "int32", "int32"],
        "ret": "uint64",
    },
    "atomic_int64_fetch_thread": {
        "symbol": "mori_shmem_atomic_int64_fetch_thread",
        "args": ["uint64", "int64", "int64", "int32", "int32", "int32"],
        "ret": "int64",
    },

    # ==== Atomic Add Convenience – Thread ====
    "uint32_atomic_add_thread": {
        "symbol": "mori_shmem_uint32_atomic_add_thread",
        "args": ["uint64", "uint32", "int32", "int32"],
        "ret": "int32",
    },
    "uint32_atomic_fetch_add_thread": {
        "symbol": "mori_shmem_uint32_atomic_fetch_add_thread",
        "args": ["uint64", "uint32", "int32", "int32"],
        "ret": "uint32",
    },
    "uint64_atomic_add_thread": {
        "symbol": "mori_shmem_uint64_atomic_add_thread",
        "args": ["uint64", "uint64", "int32", "int32"],
        "ret": "int32",
    },
    "uint64_atomic_fetch_add_thread": {
        "symbol": "mori_shmem_uint64_atomic_fetch_add_thread",
        "args": ["uint64", "uint64", "int32", "int32"],
        "ret": "uint64",
    },
    "int64_atomic_add_thread": {
        "symbol": "mori_shmem_int64_atomic_add_thread",
        "args": ["uint64", "int64", "int32", "int32"],
        "ret": "int32",
    },
    "int64_atomic_fetch_add_thread": {
        "symbol": "mori_shmem_int64_atomic_fetch_add_thread",
        "args": ["uint64", "int64", "int32", "int32"],
        "ret": "int64",
    },

    # ==== Wait ====
    "uint32_wait_until_greater_than": {
        "symbol": "mori_shmem_uint32_wait_until_greater_than",
        "args": ["uint64", "uint32"],
        "ret": "uint32",
    },
    "uint32_wait_until_equals": {
        "symbol": "mori_shmem_uint32_wait_until_equals",
        "args": ["uint64", "uint32"],
        "ret": "int32",
    },
    "uint64_wait_until_greater_than": {
        "symbol": "mori_shmem_uint64_wait_until_greater_than",
        "args": ["uint64", "uint64"],
        "ret": "uint64",
    },
    "uint64_wait_until_equals": {
        "symbol": "mori_shmem_uint64_wait_until_equals",
        "args": ["uint64", "uint64"],
        "ret": "int32",
    },
    "int32_wait_until_greater_than": {
        "symbol": "mori_shmem_int32_wait_until_greater_than",
        "args": ["uint64", "int32"],
        "ret": "int32",
    },
    "int32_wait_until_equals": {
        "symbol": "mori_shmem_int32_wait_until_equals",
        "args": ["uint64", "int32"],
        "ret": "int32",
    },
    "int64_wait_until_greater_than": {
        "symbol": "mori_shmem_int64_wait_until_greater_than",
        "args": ["uint64", "int64"],
        "ret": "int64",
    },
    "int64_wait_until_equals": {
        "symbol": "mori_shmem_int64_wait_until_equals",
        "args": ["uint64", "int64"],
        "ret": "int32",
    },
}

# Signal operation constants (matching mori::core::atomicType enum values)
SIGNAL_SET = 9
SIGNAL_ADD = 10
