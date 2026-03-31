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
"""EP Dispatch/Combine JAX test — pytest-managed multi-process version.

Run with::

    pytest tests/python/ops/test_dispatch_combine_jax.py -s -v

Environment variables:
    MORI_NUM_PROCS       — number of JAX worker processes (default: 8)
    MORI_TOTAL_GPUS      — total GPUs available (default: 8)
    MORI_SHMEM_HEAP_SIZE — shared-memory heap size  (default: 16G)
"""
import os
import multiprocessing
import socket
import traceback

import pytest

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
os.environ.setdefault("NPROC", "8")
os.environ.setdefault("PJRT_NPROC", "8")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0 \
                                    --xla_gpu_enable_command_buffer= \
                                    --xla_gpu_enable_triton_gemm=false")

_NUM_PROCS = int(os.environ.get("MORI_NUM_PROCS", "8"))
_TOTAL_GPUS = int(os.environ.get("MORI_TOTAL_GPUS", "8"))
_WORKER_TIMEOUT = int(os.environ.get("MORI_TEST_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# Multi-process infrastructure (mirrors TorchDistProcessManager for JAX)
# ---------------------------------------------------------------------------


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _jax_worker(rank, world_size, coordinator_port, total_gpus,
                task_queue, result_queue):
    """Long-lived JAX worker: init distributed + mori, then serve tasks."""
    gpus_per_proc = total_gpus // world_size
    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(
        str(rank * gpus_per_proc + i) for i in range(gpus_per_proc)
    )

    import jax
    jax.distributed.initialize(
        coordinator_address=f"localhost:{coordinator_port}",
        num_processes=world_size,
        process_id=rank,
    )

    import mori
    mori.jax.shmem_init_attr(rank, world_size)

    while True:
        task = task_queue.get()
        if task == "STOP":
            mori.jax.shmem_finalize()
            break
        func, args = task
        try:
            func(rank, world_size, *args)
            result_queue.put((rank, None))
        except Exception:
            result_queue.put((rank, traceback.format_exc()))


class JaxDistProcessManager:
    """Spawns *world_size* JAX workers that share a coordination service."""

    def __init__(self):
        self._ctx = multiprocessing.get_context("spawn")
        self.task_queue = self._ctx.Queue()
        self.result_queue = self._ctx.Queue()
        self.processes = []

    def start_workers(self, world_size, total_gpus=8):
        port = _get_free_port()
        self.processes = [
            self._ctx.Process(
                target=_jax_worker,
                args=(rank, world_size, port, total_gpus,
                      self.task_queue, self.result_queue),
            )
            for rank in range(world_size)
        ]
        for p in self.processes:
            p.start()

    def run_on_all_workers(self, func, args=None, timeout=_WORKER_TIMEOUT):
        """Submit *func(rank, world_size, \\*args)* to every worker and
        collect results.  Returns a list of ``(rank, traceback_str)`` for
        any worker that raised."""
        world_size = len(self.processes)
        for _ in range(world_size):
            self.task_queue.put((func, args or []))

        failures = []
        for _ in range(world_size):
            rank, tb = self.result_queue.get(timeout=timeout)
            if tb is not None:
                failures.append((rank, tb))
        return failures

    def shutdown(self):
        for _ in self.processes:
            self.task_queue.put("STOP")
        for p in self.processes:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()


# ---------------------------------------------------------------------------
# Session-scoped fixture — workers stay alive across all tests in the file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def jax_dist_manager():
    manager = JaxDistProcessManager()
    manager.start_workers(world_size=_NUM_PROCS, total_gpus=_TOTAL_GPUS)
    yield manager
    manager.shutdown()


# ---------------------------------------------------------------------------
# Per-worker test body (runs inside each spawned JAX process)
# ---------------------------------------------------------------------------


def _run_ep_dispatch_combine(rank, world_size,
                             hidden_dim=7168,
                             scale_dim=32,
                             max_num_inp_token_per_rank=4096,
                             num_experts_per_rank=32,
                             num_experts_per_token=56):
    """Executed in every worker — creates an op, generates data, and
    validates dispatch + combine round-trip."""
    import random
    from functools import partial

    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    try:
        from jax.shard_map import shard_map
    except ImportError:
        from jax.experimental.shard_map import shard_map
    import numpy as np
    import mori

    dtype = jnp.bfloat16
    jax_scale_dtype = jnp.float8_e4m3fnuz

    config = mori.cpp.EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=jnp.dtype(jax_scale_dtype).itemsize,
        max_token_type_size=jnp.dtype(jnp.float32).itemsize,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        warp_num_per_block=8,
        block_num=80,
        use_external_inp_buf=True,
        kernel_type=mori.cpp.EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=_TOTAL_GPUS,
        rdma_block_num=16,
        num_qp_per_pe=1,
        quant_type=mori.cpp.EpDispatchCombineQuantType.None_,
    )

    rng = jax.jit(lambda: jax.random.PRNGKey(777 + rank))()

    # -- helpers (local to this worker) ------------------------------------

    def gen_test_data(num_tokens):
        max_tokens = config.max_num_inp_token_per_rank

        def all_gather(inp):
            padded = jnp.pad(
                inp,
                [(0, max_tokens - inp.shape[0])] + [(0, 0)] * (inp.ndim - 1),
            )
            gathered = jax.lax.all_gather(padded, axis_name="i")
            return gathered.reshape(-1, *gathered.shape[2:])

        total_experts = config.num_experts_per_rank * config.world_size

        keys = jax.random.split(rng, num_tokens)
        perms = jax.vmap(
            lambda k: jax.random.permutation(k, total_experts)
        )(keys)
        indices = perms[:, :config.num_experts_per_token]
        indices_list = all_gather(indices)

        weights = jax.random.uniform(
            rng, (num_tokens, config.num_experts_per_token), dtype=jnp.float32,
        )
        weights_list = all_gather(weights)

        if config.scale_dim != 0:
            scales_fp32 = jax.random.uniform(
                rng, (num_tokens, config.scale_dim), dtype=jnp.float32,
            )
        else:
            scales_fp32 = jnp.zeros((1, 1), dtype=jnp.float32)
        scales_list = all_gather(scales_fp32).astype(jax_scale_dtype)

        input_fp32 = jax.random.normal(
            rng, (num_tokens, config.hidden_dim), dtype=jnp.float32,
        )
        input_list = all_gather(input_fp32).astype(dtype)

        print(
            f"num_tokens {num_tokens}  hidden: {config.hidden_dim} "
            f"indices: {indices.shape}/{indices.dtype}",
        )
        print(f"weights: {weights.shape}/{weights.dtype}")
        print(f"scales_fp32: {scales_fp32.shape}/{scales_fp32.dtype}", flush=True)

        return (
            indices,
            weights,
            scales_fp32.astype(jax_scale_dtype),
            input_fp32.astype(dtype),
            indices_list,
            weights_list,
            scales_list,
            input_list,
        )

    def run_test_once(op, num_tokens, test_data):
        (
            indices, weights, scales, inputs,
            indices_list, weights_list, scales_list, input_list,
        ) = test_data

        @jax.jit
        def ffi_calls(inputs, weights, scales, indices):
            (
                dispatch_output,
                dispatch_indices,
                num,
                dispatch_weights,
                dispatch_scales,
            ) = op.dispatch(inputs, weights, scales, indices)
            src_token_pos = op.get_dispatch_src_token_pos(num)

            combine_output, combine_weights = op.combine(
                dispatch_output.astype(dtype), 
                dispatch_weights, 
                dispatch_indices
            )
            return (
                (dispatch_output, dispatch_indices, num,
                 dispatch_weights, dispatch_scales),
                src_token_pos,
                combine_output,
                combine_weights,
            )

        (
            (dispatch_output, dispatch_indices, dispatch_recv_num_token,
             dispatch_weights, dispatch_scales),
            src_token_pos,
            combine_output,
            combine_weights,
        ) = ffi_calls(inputs, weights, scales, indices)

        print(f"src_token_pos: {src_token_pos} {src_token_pos.shape}")
        print(f"dispatch_recv_num_token: {dispatch_recv_num_token}")

        num_recv = dispatch_recv_num_token
        print(f"dispatch_output: {dispatch_output.shape}")
        print(
            f"rank {rank} got {num_tokens} / received {num_recv} tokens",
            flush=True,
        )

        src_num_recv_token_pos = np.array(src_token_pos)[:num_recv]
        print(
            f"src_num_recv_token_pos size: {src_num_recv_token_pos.size} "
            f"vs dispatch_recv_num_token {dispatch_recv_num_token}",
        )
        assert src_num_recv_token_pos.size == int(dispatch_recv_num_token)

        # --- validate dispatch ---
        @jax.jit
        def validate_dispatch(num, src_pos, base_list, base_out, *args):
            Y = base_list[src_pos]
            N = Y.shape[0]
            mask = jnp.arange(N) < num
            mask2D = mask[:, None]
            x = jnp.all((Y == base_out) | (~mask2D))
            for x_list, x_out in args:
                if x_out is not None:
                    x = x & jnp.all((x_list[src_pos] == x_out) | (~mask2D))
            maxv = jnp.iinfo(src_pos.dtype).max
            S_masked = jnp.where(mask, src_pos, maxv)
            S_sorted = jnp.sort(S_masked)
            eq_adjacent = S_sorted[1:] == S_sorted[:-1]
            valid = (S_sorted[1:] != maxv) & (S_sorted[:-1] != maxv)
            x = x & ~jnp.any(eq_adjacent & valid)
            return x

        res = validate_dispatch(
            dispatch_recv_num_token,
            src_token_pos,
            input_list,
            dispatch_output,
            (weights_list, dispatch_weights),
            (scales_list, dispatch_scales if config.scale_dim != 0 else None),
            (indices_list, dispatch_indices),
        )
        assert res, f"{rank} validate_dispatch failed!"
        print(f"{rank} dispatch tokens ok", flush=True)
        
        # --- validate combine ---
        @jax.jit
        def validate_combine(
            combine_output, combine_weights, inputs, weights,
            indices, num_experts_per_rank, num_tokens,
        ):
            max_tokens = combine_output.shape[0]
            mask_1d = jnp.arange(max_tokens) < num_tokens

            def masked_allclose(a, b, mask, *, atol, rtol):
                broad_mask = mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1))
                diff = jnp.abs(a - b)
                tol = atol + rtol * jnp.abs(b)
                return jnp.all((diff <= tol) | (~broad_mask))

            pes = indices // num_experts_per_rank
            pes_sorted = jnp.sort(pes, axis=-1)
            unique_pes = 1 + jnp.sum(
                pes_sorted[:, 1:] != pes_sorted[:, :-1], axis=-1,
            )

            Xinputs = inputs.astype(dtype) * unique_pes[:, None]
            inputs_buf = jnp.zeros(
                (max_tokens, Xinputs.shape[1]), dtype=Xinputs.dtype,
            )
            inputs_buf = jax.lax.dynamic_update_slice(inputs_buf, Xinputs, (0, 0))
            ok_output = masked_allclose(
                combine_output.astype(jnp.float32),
                inputs_buf.astype(jnp.float32),
                mask_1d, atol=1e-2, rtol=1e-2,
            )

            ok_weight = True
            if weights is not None:
                Xweights = weights * unique_pes[:, None]
                weights_buf = jnp.zeros(
                    (max_tokens, Xweights.shape[1]), dtype=Xweights.dtype,
                )
                weights_buf = jax.lax.dynamic_update_slice(
                    weights_buf, Xweights, (0, 0),
                )
                ok_weight = masked_allclose(
                    combine_weights, weights_buf, mask_1d, atol=1e-5, rtol=1e-5,
                )
            return ok_output & ok_weight
        print(f"{rank} running validate combine", flush=True)

        res = validate_combine(
            combine_output,
            combine_weights,
            inputs,
            weights,
            indices,
            config.num_experts_per_rank,
            num_tokens,
        )
        assert res, f"{rank} validate_combine failed!"

        print(f"{rank} combine tokens ok", flush=True)

    # -- main flow ---------------------------------------------------------

    op = mori.jax.EpDispatchCombineOp(config)
    random.seed(333 + rank)
    max_tokens = config.max_num_inp_token_per_rank
    num_tokens = int(random.randint(1, max_tokens + 1))

    devices = np.array(jax.devices())
    mesh = jax.sharding.Mesh(devices, axis_names=("i",))

    jitted_gen = jax.jit(
        shard_map(
            partial(gen_test_data, num_tokens),
            mesh=mesh,
            in_specs=(),
            out_specs=(P(), P(), P(), P(), P(), P(), P(), P()),
            check_rep=False,
        ),
        static_argnums=(),
    )

    for _ in range(1):
        test_data = jitted_gen()
        print(f"{rank} gen_test_data OK", flush=True)
        run_test_once(op, num_tokens, test_data)
    del op


# ---------------------------------------------------------------------------
# Pytest test cases
# ---------------------------------------------------------------------------


def test_ep_dispatch_combine(jax_dist_manager):
    """Run EP dispatch + combine round-trip on all workers."""
    failures = jax_dist_manager.run_on_all_workers(_run_ep_dispatch_combine)
    for r, tb in failures:
        print(f"\n=== Rank {r} FAILED ===\n{tb}")
    if failures:
        pytest.fail(
            f"{len(failures)} rank(s) failed: "
            + ", ".join(str(r) for r, _ in failures)
        )
