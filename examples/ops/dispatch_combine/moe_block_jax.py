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
"""Expert-parallel Mixture-of-Experts (MoE) block in JAX using MORI dispatch/combine.

This is a self-contained, runnable example showing how to build the
expert-parallel token exchange of a MoE layer on top of MORI's
``EpDispatchCombineOp``. It is the JAX analogue of the MORI-EP inference path
that MaxText uses (``RoutedMoE._mori_ep_matmul``).

The data flow on every rank is:

  1. Each rank owns a contiguous block of experts (``num_experts_per_rank``).
     Global expert ``e`` lives on rank ``e // num_experts_per_rank``.
  2. Top-k gate routing per token  ->  ``indices [T, topk]`` (int32) and
     ``weights [T, topk]`` (float32).
  3. ``op.dispatch`` scatters each local token to the rank(s) that own its
     selected experts, returning the received tokens padded to ``max_recv``.
  4. The per-expert SwiGLU MLP runs *locally* on the received tokens (only this
     rank's experts), masked off beyond ``num_recv`` real tokens.
  5. ``op.combine`` sums the per-rank partial results back into the original
     source-token order on the originating rank.
  6. We check the round-trip against a plain-JAX reference MoE. To keep that
     reference purely local (no collectives), every rank holds a *replicated*
     copy of all expert weights. A real model would instead shard the experts
     across ranks and keep only its own block -- dispatch/combine is exactly
     what moves the tokens to where their experts live.

MORI bakes the rank into the FFI config as a static attribute, so it needs one
GPU per process and ``jax.distributed`` + ``mori.jax.shmem_init_attr``. This
script therefore spawns one worker process per GPU; just run it directly::

    MORI_NUM_PROCS=8 MORI_TOTAL_GPUS=8 python examples/ops/dispatch_combine/moe_block_jax.py

Useful environment variables / flags (see ``--help``):
    MORI_NUM_PROCS       number of JAX worker processes  (one GPU each)
    MORI_TOTAL_GPUS      total GPUs available on the node
    MORI_SHMEM_HEAP_SIZE MORI symmetric-memory heap size (default 16G)
"""
import argparse
import multiprocessing
import os
import socket
import traceback

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
# Keep the example deterministic and avoid autotune / command-buffer noise.
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_gpu_autotune_level=0 "
    "--xla_gpu_enable_command_buffer= "
    "--xla_gpu_enable_triton_gemm=false",
)


# ---------------------------------------------------------------------------
# The MoE block (runs inside each spawned per-GPU worker process)
# ---------------------------------------------------------------------------
def run_moe_block(
    rank,
    world_size,
    hidden_dim,
    inter_dim,
    num_experts_per_rank,
    num_experts_per_token,
    num_tokens,
    total_gpus,
):
    """Build a MoE layer, run it through both a local reference and the MORI EP
    dispatch/combine path, and check that the two agree within bf16 tolerance."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import mori

    dtype = jnp.bfloat16
    total_experts = num_experts_per_rank * world_size
    assert num_experts_per_token <= total_experts, (
        f"num_experts_per_token ({num_experts_per_token}) must be <= total "
        f"experts ({total_experts})"
    )

    # -- 1. MORI op config -------------------------------------------------
    # Inference flavor: no quantization and no per-token scales
    # (scale_dim=0, quant_type=None_), intra-node kernels.
    config = mori.cpp.EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=0,
        scale_type_size=jnp.dtype(jnp.float8_e4m3fnuz).itemsize,
        max_token_type_size=jnp.dtype(jnp.float32).itemsize,
        max_num_inp_token_per_rank=num_tokens,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        warp_num_per_block=8,
        block_num=80,
        use_external_inp_buf=True,
        kernel_type=mori.cpp.EpDispatchCombineKernelType.IntraNode,
        gpu_per_node=total_gpus,
        rdma_block_num=16,
        num_qp_per_pe=1,
        quant_type=mori.cpp.EpDispatchCombineQuantType.None_,
    )
    op = mori.jax.EpDispatchCombineOp(config)

    # Upper bound on tokens this rank can receive after dispatch. Dispatch
    # outputs are always padded to this fixed size (shape-stable under jit);
    # the real count is returned separately as ``num_recv``.
    max_recv = config.max_num_tokens_to_recv()

    # -- 2a. Expert weights: REPLICATED across ranks -----------------------
    # A fixed (rank-independent) seed makes every rank generate identical expert
    # weights, so each rank holds a full copy of all experts. This is purely so
    # the reference MoE below can be computed locally without any collectives.
    # In a real model you would shard the experts and keep only your own block.
    wkey = jax.random.PRNGKey(2024)
    k0, k1, k2 = jax.random.split(wkey, 3)
    in_scale = 1.0 / np.sqrt(hidden_dim)
    out_scale = 1.0 / np.sqrt(inter_dim)
    w_gate = (
        jax.random.normal(k0, (total_experts, hidden_dim, inter_dim), jnp.float32)
        * in_scale
    ).astype(dtype)
    w_up = (
        jax.random.normal(k1, (total_experts, hidden_dim, inter_dim), jnp.float32)
        * in_scale
    ).astype(dtype)
    w_down = (
        jax.random.normal(k2, (total_experts, inter_dim, hidden_dim), jnp.float32)
        * out_scale
    ).astype(dtype)

    # -- 2b. Per-rank tokens + router logits (rank dependent) --------------
    tkey = jax.random.PRNGKey(777 + rank)
    kx, kg = jax.random.split(tkey, 2)
    x = jax.random.normal(kx, (num_tokens, hidden_dim), jnp.float32).astype(dtype)
    gate_logits = jax.random.normal(kg, (num_tokens, total_experts), jnp.float32)

    # Top-k routing: indices [T, topk] (int32), weights = softmax over the
    # selected logits [T, topk] (float32).
    topk_vals, topk_idx = jax.lax.top_k(gate_logits, num_experts_per_token)
    topk_idx = topk_idx.astype(jnp.int32)
    topk_w = jax.nn.softmax(topk_vals, axis=-1).astype(jnp.float32)

    # -- Shared per-expert MLP (SwiGLU) ------------------------------------
    def expert_fn(e, h):
        """Apply global expert ``e``'s SwiGLU MLP to rows ``h`` [N, hidden]."""
        g = h @ w_gate[e]
        u = h @ w_up[e]
        act = jax.nn.silu(g.astype(jnp.float32)).astype(dtype) * u
        return act @ w_down[e]

    # -- Reference MoE (fully local; every expert available on every rank) --
    @jax.jit
    def reference_moe(x, topk_idx, topk_w):
        out = jnp.zeros((num_tokens, hidden_dim), jnp.float32)
        for e in range(total_experts):
            sel = topk_idx == e  # [T, topk]
            w_e = jnp.sum(topk_w * sel, axis=-1)  # [T]
            out = out + w_e[:, None] * expert_fn(e, x).astype(jnp.float32)
        return out

    # -- 3-5. MORI EP path: dispatch -> local experts -> combine -----------
    @jax.jit
    def mori_moe(x, topk_idx, topk_w):
        # 3. Scatter each token to the rank(s) owning its selected experts.
        #    disp_out  [max_recv, hidden]  received tokens (padded)
        #    disp_idx  [max_recv, topk]    their selected global expert ids
        #    num_recv  scalar              real received-token count
        #    disp_w    [max_recv, topk]    their routing weights
        disp_out, disp_idx, num_recv, disp_w, _ = op.dispatch(x, topk_w, None, topk_idx)
        valid = (jnp.arange(max_recv) < num_recv)[:, None]  # [max_recv, 1]

        # 4. Run only THIS rank's experts over the received tokens, weighting
        #    each token by the routing weight of the expert that selected it.
        y = jnp.zeros((max_recv, hidden_dim), jnp.float32)
        for le in range(num_experts_per_rank):
            e = rank * num_experts_per_rank + le  # local -> global expert id
            sel = disp_idx == e  # [max_recv, topk]
            w_e = jnp.sum(disp_w * sel, axis=-1)  # [max_recv]
            y = y + w_e[:, None] * expert_fn(e, disp_out).astype(jnp.float32)
        y = (y * valid).astype(dtype)  # zero out the padded rows

        # 5. Sum partial results back to the source-token order.
        combine_out, _ = op.combine(y, None, disp_idx)
        return combine_out, num_recv

    ref = np.asarray(reference_moe(x, topk_idx, topk_w))
    mori_out, num_recv = mori_moe(x, topk_idx, topk_w)
    mori_out = np.asarray(mori_out).astype(np.float32)[:num_tokens]

    print(
        f"rank {rank}: tokens={num_tokens} experts={total_experts} "
        f"(local {num_experts_per_rank}) top_k={num_experts_per_token} "
        f"received={int(num_recv)}",
        flush=True,
    )

    # -- 6. Parity check ---------------------------------------------------
    diff = np.abs(mori_out - ref)
    tol = 2e-2 + 2e-2 * np.abs(ref)
    n_bad = int(np.sum(diff > tol))
    max_abs = float(diff.max())
    print(
        f"rank {rank}: max_abs_diff={max_abs:.4e} mismatched_elems={n_bad}"
        f" / {ref.size}",
        flush=True,
    )
    assert n_bad == 0, (
        f"rank {rank}: MORI EP MoE output differs from reference "
        f"(max_abs_diff={max_abs:.4e}, {n_bad} mismatched elements)"
    )
    print(f"rank {rank}: MoE output matches reference OK", flush=True)


# ---------------------------------------------------------------------------
# Multi-process harness: one JAX worker process per GPU
# ---------------------------------------------------------------------------
def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker(rank, world_size, coordinator_port, total_gpus, moe_kwargs, result_queue):
    """One GPU per process: init distributed + MORI shmem, run the MoE block."""
    gpus_per_proc = total_gpus // world_size
    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(
        str(rank * gpus_per_proc + i) for i in range(gpus_per_proc)
    )

    try:
        import jax

        jax.distributed.initialize(
            coordinator_address=f"localhost:{coordinator_port}",
            num_processes=world_size,
            process_id=rank,
        )

        import mori

        mori.jax.shmem_init_attr(rank, world_size)
        run_moe_block(rank, world_size, total_gpus=total_gpus, **moe_kwargs)
        result_queue.put((rank, None))
    except Exception:  # pylint: disable=broad-exception-caught
        result_queue.put((rank, traceback.format_exc()))
    finally:
        try:
            import mori

            mori.jax.shmem_finalize()
        except Exception:  # pylint: disable=broad-exception-caught
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--world_size",
        type=int,
        default=int(os.environ.get("MORI_NUM_PROCS", "8")),
        help="Number of worker processes (one GPU each).",
    )
    parser.add_argument(
        "--total_gpus",
        type=int,
        default=int(os.environ.get("MORI_TOTAL_GPUS", "8")),
        help="Total GPUs available on the node.",
    )
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--inter_dim", type=int, default=2048)
    parser.add_argument("--num_experts_per_rank", type=int, default=2)
    parser.add_argument("--num_experts_per_token", type=int, default=4)
    parser.add_argument("--num_tokens", type=int, default=256)
    parser.add_argument(
        "--timeout", type=int, default=int(os.environ.get("MORI_TEST_TIMEOUT", "300"))
    )
    args = parser.parse_args()

    moe_kwargs = dict(
        hidden_dim=args.hidden_dim,
        inter_dim=args.inter_dim,
        num_experts_per_rank=args.num_experts_per_rank,
        num_experts_per_token=args.num_experts_per_token,
        num_tokens=args.num_tokens,
    )

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    port = _get_free_port()
    procs = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                args.world_size,
                port,
                args.total_gpus,
                moe_kwargs,
                result_queue,
            ),
        )
        for rank in range(args.world_size)
    ]
    for p in procs:
        p.start()

    failures = []
    for _ in range(args.world_size):
        rank, tb = result_queue.get(timeout=args.timeout)
        if tb is not None:
            failures.append((rank, tb))
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()

    for r, tb in failures:
        print(f"\n=== Rank {r} FAILED ===\n{tb}")
    if failures:
        raise SystemExit(
            f"{len(failures)} rank(s) failed: " + ", ".join(str(r) for r, _ in failures)
        )
    print("All ranks: MORI EP MoE block matches reference OK", flush=True)


if __name__ == "__main__":
    main()
