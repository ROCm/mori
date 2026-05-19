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

Validation strategy: each thread generates per-rank inputs deterministically
from PRNGKey(BASE_SEED + rank). Because every thread runs the same generator
function, each one can locally re-seed for every other rank to reconstruct the
full input_list — no cross-thread/cross-process all-gather needed. We then
mirror the validate_dispatch / validate_combine logic from
test_dispatch_combine_jax.py to verify dispatched data matches sources and
combined output equals input * unique_pes.

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

# Match the env-var bypass set by the multi-process JAX EP test:
# - MORI_SHMEM_HEAP_SIZE=16G: default 4G is tight for world_size=8 EP buffers.
# - XLA_FLAGS:
#     --xla_gpu_autotune_level=0      → skip autotune (slow on first JIT)
#     --xla_gpu_enable_command_buffer= → disable HIP command buffer (graph)
#     --xla_gpu_enable_triton_gemm=false → avoid Triton-AMDGPU pass errors
#                                          ("TritonAMDGPUMoveUpPrologueLoads")
# Use setdefault so user-supplied values still win.
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_autotune_level=0 "
    "--xla_gpu_enable_command_buffer= "
    "--xla_gpu_enable_triton_gemm=false",
)


# Shared PRNG seed base. All threads use PRNGKey(BASE_SEED + rank) so any
# thread can reconstruct any other rank's inputs by re-seeding.
BASE_SEED = 123
NUM_TOKENS_PER_RANK = 32


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


def _spmt_shmem_init_one_thread(rank, world_size, unique_id):
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
    """Build an EP config. ``gpu_per_node`` is the per-node PE count (NOT the
    physical GPU count of the box). For single-node SPMT testing pass
    ``gpu_per_node = world_size``; the EP handle asserts
    IsPowerOf2(gpuPerNode) && worldSize % gpuPerNode == 0."""
    import jax.numpy as jnp
    import mori

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


def _gen_per_rank_inputs(rank, config, num_tokens):
    """Deterministic per-rank input generation. Every thread can reproduce any
    rank's inputs by passing that rank's index here.

    Returns (indices, weights, inputs) as JAX arrays on CPU (caller does
    device_put as needed). Shapes:
      indices: (num_tokens, num_experts_per_token)   int32
      weights: (num_tokens, num_experts_per_token)   float32
      inputs:  (num_tokens, hidden_dim)              bfloat16
    """
    import jax
    import jax.numpy as jnp

    rng = jax.random.PRNGKey(BASE_SEED + rank)
    total_experts = config.num_experts_per_rank * config.world_size

    keys = jax.random.split(rng, num_tokens)
    indices = jax.vmap(lambda k: jax.random.permutation(k, total_experts))(keys)[
        :, : config.num_experts_per_token
    ].astype(jnp.int32)
    weights = jax.random.uniform(
        rng, (num_tokens, config.num_experts_per_token), dtype=jnp.float32
    )
    inputs = jax.random.normal(
        rng, (num_tokens, config.hidden_dim), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    return indices, weights, inputs


def _build_full_input_lists(world_size, config, num_tokens):
    """Reconstruct every rank's inputs locally (no cross-thread comm needed)
    and concatenate into world_size * max_num_inp_token_per_rank padded lists,
    matching the layout that multi-process test produces via jax.lax.all_gather.
    """
    import jax.numpy as jnp

    max_tokens = config.max_num_inp_token_per_rank
    indices_list, weights_list, inputs_list = [], [], []
    for r in range(world_size):
        ind, wt, inp = _gen_per_rank_inputs(r, config, num_tokens)
        # Pad each rank's contribution to max_tokens to match all_gather layout.
        pad = max_tokens - num_tokens
        if pad > 0:
            ind = jnp.pad(ind, [(0, pad), (0, 0)])
            wt = jnp.pad(wt, [(0, pad), (0, 0)])
            inp = jnp.pad(inp, [(0, pad), (0, 0)])
        indices_list.append(ind)
        weights_list.append(wt)
        inputs_list.append(inp)
    return (
        jnp.concatenate(indices_list, axis=0),
        jnp.concatenate(weights_list, axis=0),
        jnp.concatenate(inputs_list, axis=0),
    )


def _validate_dispatch(
    num, src_pos, tok_stride, inp_tok_per_rank, base_list, base_out, *args
):
    """Mirror of validate_dispatch from test_dispatch_combine_jax.py.

    For each received token, decode (sender_pe, local_tok_id) from src_pos,
    look up the original input via base_list[pe * inp_tok_per_rank + local_id],
    and check it matches the dispatched output. Also check that no two received
    tokens share the same src_pos (no double-delivery).
    """
    import jax.numpy as jnp

    pe = src_pos // tok_stride
    local_tok_id = src_pos - pe * tok_stride
    list_idx = pe * inp_tok_per_rank + local_tok_id
    Y = base_list[list_idx]
    N = Y.shape[0]
    mask = jnp.arange(N) < num
    mask2D = mask[:, None]
    x = jnp.all((Y == base_out) | (~mask2D))
    for x_list, x_out in args:
        if x_out is not None:
            x = x & jnp.all((x_list[list_idx] == x_out) | (~mask2D))
    maxv = jnp.iinfo(src_pos.dtype).max
    s_masked = jnp.where(mask, src_pos, maxv)
    s_sorted = jnp.sort(s_masked)
    eq_adjacent = s_sorted[1:] == s_sorted[:-1]
    valid = (s_sorted[1:] != maxv) & (s_sorted[:-1] != maxv)
    x = x & ~jnp.any(eq_adjacent & valid)
    return x


def _validate_combine(
    combine_output,
    combine_weights,
    inputs,
    weights,
    indices,
    num_experts_per_rank,
    num_tokens,
    dtype,
):
    """Mirror of validate_combine from test_dispatch_combine_jax.py.

    Each input token is dispatched to `unique_pes` distinct PEs; combine
    sums the `unique_pes` copies, so combined output should equal
    `input * unique_pes` (and combined weights = `weights * unique_pes`).
    Uses bf16-tolerant atol/rtol on the output and tight tolerance on weights.
    """
    import jax
    import jax.numpy as jnp

    max_tokens = combine_output.shape[0]
    mask_1d = jnp.arange(max_tokens) < num_tokens

    def masked_allclose(a, b, mask, *, atol, rtol):
        broad_mask = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
        diff = jnp.abs(a - b)
        tol = atol + rtol * jnp.abs(b)
        return jnp.all((diff <= tol) | (~broad_mask))

    pes = indices // num_experts_per_rank
    pes_sorted = jnp.sort(pes, axis=-1)
    unique_pes = 1 + jnp.sum(pes_sorted[:, 1:] != pes_sorted[:, :-1], axis=-1)

    x_inputs = inputs.astype(dtype) * unique_pes[:, None]
    inputs_buf = jnp.zeros((max_tokens, x_inputs.shape[1]), dtype=x_inputs.dtype)
    inputs_buf = jax.lax.dynamic_update_slice(inputs_buf, x_inputs, (0, 0))
    ok_output = masked_allclose(
        combine_output.astype(jnp.float32),
        inputs_buf.astype(jnp.float32),
        mask_1d,
        atol=1e-2,
        rtol=1e-2,
    )

    ok_weight = True
    if weights is not None:
        x_weights = weights * unique_pes[:, None]
        weights_buf = jnp.zeros((max_tokens, x_weights.shape[1]), dtype=x_weights.dtype)
        weights_buf = jax.lax.dynamic_update_slice(weights_buf, x_weights, (0, 0))
        ok_weight = masked_allclose(
            combine_weights,
            weights_buf,
            mask_1d,
            atol=1e-5,
            rtol=1e-5,
        )
    return ok_output & ok_weight


def _ep_thread_body(rank, world_size, unique_id, results):
    """Per-thread body: init shmem + run EP dispatch+combine + verify data."""
    err = None
    try:
        _spmt_shmem_init_one_thread(rank, world_size, unique_id)

        import gc

        import jax
        import jax.numpy as jnp
        import mori
        import numpy as np
        from mori import cpp, shmem

        # gpu_per_node = world_size for single-node SPMT (see _build_config).
        config = _build_config(rank, world_size, gpu_per_node=world_size)
        op = mori.jax.EpDispatchCombineOp(config)

        my_dev = jax.devices()[rank]
        num_tokens = NUM_TOKENS_PER_RANK
        dtype = jnp.bfloat16

        # --- per-rank inputs (this thread's) ---
        indices, weights, inputs = _gen_per_rank_inputs(rank, config, num_tokens)
        indices = jax.device_put(indices, my_dev)
        weights = jax.device_put(weights, my_dev)
        inputs = jax.device_put(inputs, my_dev)

        # --- full inputs_list rebuilt locally on this device ---
        # (every rank generates the same content from PRNGKey(BASE_SEED + r))
        indices_list, weights_list, inputs_list = _build_full_input_lists(
            world_size, config, num_tokens
        )
        indices_list = jax.device_put(indices_list, my_dev)
        weights_list = jax.device_put(weights_list, my_dev)
        inputs_list = jax.device_put(inputs_list, my_dev)

        # --- run dispatch + get src token positions ---
        with jax.default_device(my_dev):
            (
                dispatch_output,
                dispatch_indices,
                dispatch_recv_num_token,
                dispatch_weights,
                _scales,
            ) = op.dispatch(inputs, weights, None, indices)
            src_token_pos = op.get_dispatch_src_token_pos(dispatch_recv_num_token)

            num_recv = int(np.asarray(dispatch_recv_num_token))
            print(f"[thread {rank}] dispatched, recv {num_recv} tokens", flush=True)

            # Sanity: src_token_pos length matches num_recv.
            src_arr = np.asarray(src_token_pos)[:num_recv]
            assert src_arr.size == num_recv, (
                f"rank {rank}: src_token_pos size {src_arr.size} "
                f"!= dispatch_recv_num_token {num_recv}"
            )

            # --- validate dispatch: dispatched tokens match sources ---
            tok_stride = config.max_num_tokens_to_send()
            inp_tok_per_rank = config.max_num_inp_token_per_rank
            ok_dispatch = _validate_dispatch(
                dispatch_recv_num_token,
                src_token_pos,
                tok_stride,
                inp_tok_per_rank,
                inputs_list,
                dispatch_output,
                (weights_list, dispatch_weights),
                (indices_list, dispatch_indices),
            )
            assert bool(
                np.asarray(ok_dispatch)
            ), f"rank {rank} validate_dispatch FAILED"
            print(f"[thread {rank}] dispatch data verified", flush=True)

            # --- run combine ---
            combine_out, combine_w = op.combine(
                dispatch_output.astype(dtype),
                dispatch_weights,
                dispatch_indices,
            )

            # --- validate combine: output == input * unique_pes ---
            ok_combine = _validate_combine(
                combine_out,
                combine_w,
                inputs,
                weights,
                indices,
                config.num_experts_per_rank,
                num_tokens,
                dtype,
            )
            assert bool(np.asarray(ok_combine)), f"rank {rank} validate_combine FAILED"
            print(f"[thread {rank}] combine data verified", flush=True)

        del op

        # Release handle references before shmem_finalize frees the
        # underlying shmem buffers.  jax.clear_caches() is process-global
        # and racy across SPMT threads — avoid it.
        cpp.clear_ep_handle_cache()
        gc.collect()
        shmem.shmem_finalize()

    except Exception:
        err = traceback.format_exc()

    results[rank] = err


def _run_spmt(world_size: int, kernel_dir: str):
    num_gpus = _get_num_gpus()
    if num_gpus < world_size:
        pytest.skip(f"Need {world_size} GPUs, only {num_gpus} available")

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
            args=(rank, world_size, unique_id, results),
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
    """Each world_size runs in an isolated subprocess to ensure clean
    shmem_init / shmem_finalize lifecycle per test case."""
    import subprocess
    import sys

    kernel_dir = os.environ.get("MORI_KERNEL_DIR", "")
    if not kernel_dir or not os.path.isdir(kernel_dir):
        pytest.skip(
            "MORI_KERNEL_DIR must point to a directory of AOT-compiled .hsaco "
            "(BUILD_OPS_DEVICE=ON build artifacts)."
        )

    env = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from tests.python.ops.test_dispatch_combine_jax_spmt "
            f"import _run_spmt; "
            f"_run_spmt({world_size}, {kernel_dir!r})",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        close_fds=True,
        cwd=os.environ.get("PYTHONPATH", "").split(":")[0] or ".",
    )
    if result.returncode != 0:
        out = (result.stdout or "")[-2000:]
        err = (result.stderr or "")[-2000:]
        print(out)
        print(err)
    assert (
        result.returncode == 0
    ), f"Subprocess for world_size={world_size} failed (rc={result.returncode})"
