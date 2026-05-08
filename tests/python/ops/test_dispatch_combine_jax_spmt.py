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
"""SPMT (Single-Process Multi-Thread) EP smoke test for JAX.

Spawns N host threads inside a single Python process.  Each thread binds to
its own GPU via hipSetDevice and drives MORI EP independently — no
multiprocessing, no jax.distributed.initialize.  This is the JAX-on-SPMD model
that the JAX team needs but that the existing test_dispatch_combine_jax.py
(multi-process) does not exercise.

Requirements:
  - mori built with MORI_MULTITHREAD_SUPPORT=ON, BUILD_OPS_DEVICE=ON,
    BUILD_XLA_FFI_OPS=ON
  - MORI_KERNEL_DIR pointing to AOT-compiled .hsaco directory
  - At least 2 GPUs visible (do NOT set HIP_VISIBLE_DEVICES to a subset)
"""
import ctypes
import os
import threading
import traceback

import pytest


def _get_num_gpus() -> int:
    """Query HIP for visible device count without importing torch."""
    from mori.jit.hip_driver import _get_hip_lib

    hip = _get_hip_lib()
    n = ctypes.c_int(0)
    err = hip.hipGetDeviceCount(ctypes.byref(n))
    if err != 0:
        return 0
    return int(n.value)


def _hip_set_device(dev: int) -> None:
    from mori.jit.hip_driver import _get_hip_lib

    hip = _get_hip_lib()
    err = hip.hipSetDevice(ctypes.c_int(dev))
    if err != 0:
        raise RuntimeError(f"hipSetDevice({dev}) failed: {err}")


def _spmt_shmem_init_one_thread(rank, world_size, unique_id, kernel_dir):
    """Init MORI shmem for one rank inside an SPMT thread.

    Bypasses mori.jax.shmem_init_attr (which requires jax.distributed client)
    and calls the underlying mori.shmem APIs directly.
    """
    from mori import cpp, shmem

    _hip_set_device(rank)
    shmem.shmem_init_attr(
        shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world_size, unique_id
    )
    # Preload AOT EP kernels into THIS thread's GPU's HIP context.
    cpp.preload_kernels()


def _build_config(rank, world_size, gpu_per_node):
    import mori
    import jax.numpy as jnp

    return mori.cpp.EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        hidden_dim=2048,
        scale_dim=0,
        scale_type_size=1,
        max_token_type_size=jnp.dtype(jnp.float32).itemsize,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=8,
        num_experts_per_token=4,
        warp_num_per_block=8,
        block_num=80,
        use_external_inp_buf=True,
        kernel_type=mori.cpp.EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=gpu_per_node,
        rdma_block_num=16,
        num_qp_per_pe=1,
        quant_type=mori.cpp.EpDispatchCombineQuantType.None_,
    )


def _ep_thread_body(rank, world_size, unique_id, kernel_dir, results):
    """Per-thread body: init shmem + create EP op + run dispatch round-trip."""
    err = None
    try:
        _spmt_shmem_init_one_thread(rank, world_size, unique_id, kernel_dir)

        import jax
        import jax.numpy as jnp
        import numpy as np
        import mori
        from mori import cpp, shmem

        config = _build_config(rank, world_size, gpu_per_node=world_size)
        op = mori.jax.EpDispatchCombineOp(config)

        # Build per-rank inputs on this thread's device.
        # Use jax.device_put to ensure data is on rank's GPU.
        my_dev = jax.devices()[rank]

        rng = jax.random.PRNGKey(123 + rank)
        num_tokens = 32

        total_experts = config.num_experts_per_rank * config.world_size
        keys = jax.random.split(rng, num_tokens)
        indices = jax.vmap(
            lambda k: jax.random.permutation(k, total_experts)
        )(keys)[:, : config.num_experts_per_token].astype(jnp.int32)
        weights = jax.random.uniform(
            rng, (num_tokens, config.num_experts_per_token), dtype=jnp.float32
        )
        inputs = jax.random.normal(
            rng, (num_tokens, config.hidden_dim), dtype=jnp.float32
        ).astype(jnp.bfloat16)

        # Place inputs on this thread's device.
        indices = jax.device_put(indices, my_dev)
        weights = jax.device_put(weights, my_dev)
        inputs = jax.device_put(inputs, my_dev)

        # Run dispatch on this device.
        with jax.default_device(my_dev):
            (
                dispatch_output,
                dispatch_indices,
                dispatch_recv_num_token,
                dispatch_weights,
                _,
            ) = op.dispatch(inputs, weights, None, indices)

            # Force materialization to make sure FFI completes
            num_recv = int(np.asarray(dispatch_recv_num_token))
            print(
                f"[thread {rank}] dispatch OK, recv {num_recv} tokens",
                flush=True,
            )

            combine_out, combine_w = op.combine(
                dispatch_output.astype(jnp.bfloat16),
                dispatch_weights,
                dispatch_indices,
            )
            # Materialize.
            _ = np.asarray(combine_out[:1])
            print(f"[thread {rank}] combine OK", flush=True)

        del op

        # Per-thread cleanup so the next test (different world_size) can
        # re-init this slot without conflict.
        cpp.clear_ep_handle_cache()
        shmem.shmem_finalize()

    except Exception:
        err = traceback.format_exc()

    results[rank] = err


def _run_spmt(world_size: int, kernel_dir: str):
    if _get_num_gpus() < world_size:
        pytest.skip(f"Need {world_size} GPUs")

    # Each thread binds to a different device → don't pre-set HIP_VISIBLE_DEVICES
    os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")
    os.environ.setdefault("MORI_KERNEL_DIR", kernel_dir)

    # Generate unique_id from main thread (rank 0 will publish).
    from mori import shmem

    unique_id = shmem.shmem_get_unique_id()

    results = [None] * world_size
    threads = [
        threading.Thread(
            target=_ep_thread_body,
            args=(rank, world_size, unique_id, kernel_dir, results),
            daemon=True,
            name=f"ep-spmt-{rank}",
        )
        for rank in range(world_size)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
        assert not t.is_alive(), f"Thread {t.name} timed out"

    for rank, err in enumerate(results):
        if err is not None:
            print(f"\n=== Thread {rank} FAILED ===\n{err}\n")
    failed = [r for r, e in enumerate(results) if e is not None]
    assert not failed, f"Failed threads: {failed}"


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_jax_ep_spmt(world_size):
    kernel_dir = os.environ.get("MORI_KERNEL_DIR", "")
    if not kernel_dir or not os.path.isdir(kernel_dir):
        pytest.skip(
            "MORI_KERNEL_DIR must point to a directory of AOT-compiled .hsaco "
            "(BUILD_OPS_DEVICE=ON build artifacts)."
        )
    _run_spmt(world_size, kernel_dir)
