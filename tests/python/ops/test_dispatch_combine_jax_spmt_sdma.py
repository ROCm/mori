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
"""SPMT (Single-Process Multi-Thread) AsyncLL+SDMA EP smoke test for JAX.

Exercises the SDMA transport path with the AsyncLL kernel type.  Structure
mirrors test_dispatch_combine_jax_spmt.py (IntraNode) but sets
MORI_ENABLE_SDMA=1 and uses KernelType.AsyncLL so that the SDMA signal
exchange, SDMA put/quiet, and the split send/recv kernel sequence are all
covered end-to-end.

Requirements:
  - mori built with MORI_MULTITHREAD_SUPPORT=ON, BUILD_OPS_DEVICE=ON,
    BUILD_XLA_FFI_OPS=ON
  - MORI_KERNEL_DIR pointing to AOT-compiled .hsaco directory
    (must include ep_async_ll kernels)
  - At least 2 GPUs visible with SDMA / peer-access capability
"""

import ctypes
import os
import threading
import traceback

import pytest

BASE_SEED = 456
NUM_TOKENS_PER_RANK = 32


def _get_num_gpus() -> int:
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
    from mori import cpp, shmem

    _hip_set_device(rank)
    shmem.shmem_init_attr(
        shmem.MORI_SHMEM_INIT_WITH_UNIQUEID, rank, world_size, unique_id
    )
    cpp.preload_kernels()


def _build_config(rank, world_size, gpu_per_node):
    import jax.numpy as jnp
    import mori

    return mori.cpp.EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        hidden_dim=4096,
        scale_dim=0,
        scale_type_size=1,
        max_token_type_size=jnp.dtype(jnp.float32).itemsize,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=8,
        num_experts_per_token=4,
        warp_num_per_block=8,
        block_num=64,
        use_external_inp_buf=True,
        kernel_type=mori.cpp.EpDispatchCombineKernelType.AsyncLL,
        gpu_per_node=gpu_per_node,
        rdma_block_num=16,
        num_qp_per_pe=1,
        quant_type=mori.cpp.EpDispatchCombineQuantType.None_,
    )


def _gen_per_rank_inputs(rank, config, num_tokens):
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
    import jax.numpy as jnp

    max_tokens = config.max_num_inp_token_per_rank
    indices_list, weights_list, inputs_list = [], [], []
    for r in range(world_size):
        ind, wt, inp = _gen_per_rank_inputs(r, config, num_tokens)
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
    if combine_weights is not None and weights is not None:
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
    err = None
    try:
        _spmt_shmem_init_one_thread(rank, world_size, unique_id)

        import gc

        import jax
        import jax.numpy as jnp
        import mori
        import numpy as np
        from mori import cpp, shmem

        config = _build_config(rank, world_size, gpu_per_node=world_size)
        op = mori.jax.EpDispatchCombineOp(config)

        my_dev = jax.devices()[rank]
        num_tokens = NUM_TOKENS_PER_RANK
        dtype = jnp.bfloat16

        indices, weights, inputs = _gen_per_rank_inputs(rank, config, num_tokens)
        indices = jax.device_put(indices, my_dev)
        weights = jax.device_put(weights, my_dev)
        inputs = jax.device_put(inputs, my_dev)

        indices_list, weights_list, inputs_list = _build_full_input_lists(
            world_size, config, num_tokens
        )
        indices_list = jax.device_put(indices_list, my_dev)
        weights_list = jax.device_put(weights_list, my_dev)
        inputs_list = jax.device_put(inputs_list, my_dev)

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
            print(
                f"[sdma-thread {rank}] dispatched, recv {num_recv} tokens", flush=True
            )

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
            print(f"[sdma-thread {rank}] dispatch data verified", flush=True)

            combine_out, combine_w = op.combine(
                dispatch_output.astype(dtype),
                None,
                dispatch_indices,
            )

            ok_combine = _validate_combine(
                combine_out,
                None,
                inputs,
                weights,
                indices,
                config.num_experts_per_rank,
                num_tokens,
                dtype,
            )
            assert bool(np.asarray(ok_combine)), f"rank {rank} validate_combine FAILED"
            print(f"[sdma-thread {rank}] combine data verified", flush=True)

        del op
        cpp.clear_ep_handle_cache()
        gc.collect()
        shmem.shmem_finalize()

    except Exception:
        err = traceback.format_exc()

    results[rank] = err


def _run_spmt_sdma(world_size: int, kernel_dir: str):
    num_gpus = _get_num_gpus()
    if num_gpus < world_size:
        pytest.skip(f"Need {world_size} GPUs, only {num_gpus} available")

    os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")
    os.environ.setdefault("MORI_KERNEL_DIR", kernel_dir)

    from mori import shmem

    unique_id = shmem.shmem_get_unique_id()

    results = [None] * world_size
    threads = [
        threading.Thread(
            target=_ep_thread_body,
            args=(rank, world_size, unique_id, results),
            daemon=True,
            name=f"ep-spmt-sdma-{rank}",
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
def test_jax_ep_spmt_sdma(world_size):
    """Each world_size runs in an isolated subprocess.

    Each parametrized case needs a clean shmem_init/shmem_finalize lifecycle.
    The AnvilLib singleton and KFD SDMA queues are not fully released by
    shmem_finalize, so process isolation ensures OS-level cleanup between cases.
    """
    import subprocess
    import sys

    kernel_dir = os.environ.get("MORI_KERNEL_DIR", "")
    if not kernel_dir or not os.path.isdir(kernel_dir):
        pytest.skip(
            "MORI_KERNEL_DIR must point to a directory of AOT-compiled .hsaco "
            "(BUILD_OPS_DEVICE=ON build artifacts)."
        )

    env = os.environ.copy()
    env["MORI_ENABLE_SDMA"] = "1"
    env.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
    env.setdefault(
        "XLA_FLAGS",
        "--xla_gpu_autotune_level=0 "
        "--xla_gpu_enable_command_buffer= "
        "--xla_gpu_enable_triton_gemm=false",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from tests.python.ops.test_dispatch_combine_jax_spmt_sdma "
            f"import _run_spmt_sdma; "
            f"_run_spmt_sdma({world_size}, {kernel_dir!r})",
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
        err = "\n".join(
            l for l in (result.stderr or "").splitlines() if "libibverbs" not in l
        )[-2000:]
        print(out)
        print(err)
    assert (
        result.returncode == 0
    ), f"Subprocess for world_size={world_size} failed (rc={result.returncode})"
