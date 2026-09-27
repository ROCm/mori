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
"""EP2/EP4 top-k: exact row-sensitive reference, ragged/dropped routes and replay.

Enable MORI_RDNA4_EP=1, MORI_EP_COMM=shmem, MORI_SHMEM_MODE=static_heap.
Select 2 or 4 idle gfx1201 GPUs and set VERIFY_WORLD_SIZE accordingly.
VERIFY_TOPKS accepts a comma list or "all" for every integer from 1 to 64.
This is correctness-only; reference construction and transformations are not timed.
"""

import gc
from itertools import product
import os
import traceback
import torch
import mori
from tests.python.ops.dispatch_combine_test_utils import (
    assert_worker_results,
    start_torch_dist_process_manager,
)

WS = int(os.environ.get("VERIFY_WORLD_SIZE", "2"))
assert WS in (2, 4)
EXPERTS_PER_RANK = 256 // WS


def run_worker(rank):
    device = torch.device("cuda", rank)
    hiddens = [
        int(x) for x in os.environ.get("VERIFY_HIDDEN_SIZES", "2056,8192").split(",")
    ]
    capacities = [
        int(x)
        for x in os.environ.get(
            "VERIFY_CAPACITIES", os.environ.get("VERIFY_CAPACITY", "512")
        ).split(",")
    ]
    topks_env = os.environ.get("VERIFY_TOPKS", "1,2,4,6,8,9,16,31,32,33,48,63,64")
    topks = (
        list(range(1, 65))
        if topks_env == "all"
        else [int(k) for k in topks_env.split(",")]
    )
    for dtype, hidden, capacity, topk in product(
        (torch.bfloat16, torch.float16), hiddens, capacities, topks
    ):
        cfg = mori.ops.EpDispatchCombineConfig(
            data_type=dtype,
            rank=rank,
            world_size=WS,
            gpu_per_node=WS,
            hidden_dim=hidden,
            scale_dim=0,
            scale_type_size=1,
            max_token_type_size=2,
            max_num_inp_token_per_rank=capacity,
            num_experts_per_rank=EXPERTS_PER_RANK,
            num_experts_per_token=topk,
            block_num=32,
            warp_num_per_block=8,
            use_external_inp_buf=True,
            kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
        )
        op = mori.ops.EpDispatchCombineOp(cfg)
        for case, (counts, route) in enumerate(
            (
                [
                    ((0, 17), "random"),
                    ((19, 1), "local"),
                    ((257, 257), "remote"),
                    ((31, 47), "invalid"),
                    ((0, 0), "random"),
                    ((7, 11), "dropped"),
                    # Cross the small-entry boundary independently on each rank.
                    ((64, 257), "random"),
                    ((257, 128), "random"),
                    ((256, 256), "random"),
                    ((192, 64), "remote"),
                    ((33, 257), "last_slot"),
                    ((65, 127), "split_wave"),
                    ((33, 257), "single_dest"),
                ]
                if WS == 2
                else [
                    ((0, 17, 7, 0), "random"),
                    ((19, 1, 7, 3), "local"),
                    ((257, 257, 257, 257), "remote"),
                    ((31, 47, 15, 11), "invalid"),
                    ((0, 0, 0, 0), "random"),
                    ((7, 11, 5, 19), "dropped"),
                    ((64, 257, 0, 192), "random"),
                    ((257, 128, 64, 17), "random"),
                    ((256, 256, 256, 256), "random"),
                    ((192, 64, 128, 257), "remote"),
                    ((33, 257, 17, 1), "last_slot"),
                    ((65, 127, 19, 71), "split_wave"),
                    ((33, 257, 17, 1), "single_dest"),
                ]
            )
        ):
            if max(counts) > cfg.max_num_inp_token_per_rank:
                continue
            inputs, indices, weights, memberships = [], [], [], []
            use_weights = case % 2 == 0 or route == "split_wave"
            print(
                f"rank={rank} start {dtype} H={hidden} K={topk} capacity={capacity} case={case}",
                flush=True,
            )
            for source in range(WS):
                n = counts[source]
                rng = torch.Generator(device=device).manual_seed(
                    7331 + source + 17 * case
                )
                # Finite values spanning signs and many exponents; not just dyadic integers.
                x = torch.randn((n, hidden), generator=rng, device=device)
                exponent = torch.randint(
                    -8, 8, (n, hidden), generator=rng, device=device
                )
                inputs.append((x * torch.exp2(exponent.float())).to(dtype))
                idx = torch.randint(
                    0, 256, (n, topk), generator=rng, device=device, dtype=torch.int32
                )
                if route == "local":
                    idx = idx % EXPERTS_PER_RANK + EXPERTS_PER_RANK * source
                elif route == "remote":
                    idx = idx % EXPERTS_PER_RANK + EXPERTS_PER_RANK * (
                        (source + 1) % WS
                    )
                elif route == "invalid":
                    idx[:, 0::3] = -1
                    idx[:, 1::3] = 256
                    if n:
                        idx[0] = -1
                elif route == "dropped":
                    idx.fill_(-1)
                elif route == "single_dest":
                    idx.fill_(0)
                elif route in ("last_slot", "split_wave"):
                    idx.fill_(-1)
                    if route == "split_wave":
                        idx[:, 0] = source * EXPERTS_PER_RANK
                        if topk > 33:
                            # Duplicate the local destination across the wave boundary.
                            idx[:, 32] = idx[:, 0]
                    # For K>32 this destination appears only in the second batch.
                    idx[:, -1] = ((source + 1) % WS) * EXPERTS_PER_RANK
                indices.append(idx)
                memberships.append(
                    torch.stack(
                        [
                            (
                                (idx >= p * EXPERTS_PER_RANK)
                                & (idx < (p + 1) * EXPERTS_PER_RANK)
                            ).any(1)
                            for p in range(WS)
                        ],
                        dim=1,
                    )
                )
                weights.append(torch.randn((n, topk), generator=rng, device=device))

            def check_dispatch(operator, out, source_inputs, routing=None):
                received = int(out[4].item())
                assert received == sum(int(m[:, rank].sum()) for m in memberships)
                reverse = (
                    routing.disp_tok_id_to_src_tok_id_local
                    if routing is not None
                    else operator.get_dispatch_src_token_pos()
                )[:received].long()
                stride = operator.max_num_tokens_to_send()
                assert reverse.unique().numel() == received
                for source in range(WS):
                    mask = reverse // stride == source
                    rows = reverse[mask] % stride
                    assert bool((rows < counts[source]).all())
                    assert bool(memberships[source][rows, rank].all())
                    torch.testing.assert_close(
                        out[0][:received][mask],
                        source_inputs[source][rows],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        out[3][:received][mask],
                        indices[source][rows],
                        rtol=0,
                        atol=0,
                    )
                    if use_weights:
                        torch.testing.assert_close(
                            out[1][:received][mask],
                            weights[source][rows],
                            rtol=0,
                            atol=0,
                        )

            # The existing routing-handle API requires a positive source
            # count. Empty-rank tests use the ordinary (non-handle) API.
            use_handle = (
                min(counts) > 0 and os.environ.get("VERIFY_ROUTING_HANDLES", "1") != "0"
            )
            out = op.dispatch(
                inputs[rank],
                weights[rank] if use_weights else None,
                None,
                indices[rank],
                return_routing=use_handle,
            )
            handle = out[5] if use_handle else None
            check_dispatch(op, out, inputs, handle)
            saved_map = handle.disp_dest_tok_id_map.clone() if use_handle else None
            saved_count = handle.total_recv_token_num.clone() if use_handle else None
            if use_handle:
                # Reuse dispatch staging with no intervening combine.
                out = op.dispatch(
                    inputs[rank],
                    weights[rank] if use_weights else None,
                    None,
                    indices[rank],
                    routing=handle,
                )
                check_dispatch(op, out, inputs, handle)
            for replay in (False, True) if use_handle else (False,):
                source_inputs = [x.neg() if replay else x for x in inputs]
                if replay:
                    if os.environ.get("VERIFY_HANDLE_INTERLEAVE") == "1":
                        # Replay an older handle after a different routing
                        # call overwrites this operator's transient counts.
                        other_indices = (indices[rank] + EXPERTS_PER_RANK).remainder(
                            256
                        )
                        other = op.dispatch(
                            inputs[rank],
                            weights[rank] if use_weights else None,
                            None,
                            other_indices,
                        )
                        op.combine(
                            other[0],
                            other[1] if use_weights else None,
                            other_indices,
                        )
                    out = op.dispatch(
                        source_inputs[rank],
                        weights[rank] if use_weights else None,
                        None,
                        indices[rank],
                        routing=handle,
                    )
                    check_dispatch(op, out, source_inputs, handle)
                    torch.testing.assert_close(
                        handle.disp_dest_tok_id_map, saved_map, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        handle.total_recv_token_num, saved_count, rtol=0, atol=0
                    )
                received = int(out[4].item())
                # Distinct, row-dependent expert output per rank. Explicit rounding matches
                # the reference and tests local/remote source selection independently.
                expert = (out[0][:received] * (rank + 1) + (rank + 1) / 7).to(dtype)
                output, out_weights = op.combine(
                    expert,
                    out[1][:received] if use_weights else None,
                    indices[rank],
                    routing=handle,
                )
                first_output = first_weights = None
                sign = 1
                if use_handle and os.environ.get("VERIFY_DOUBLE_COMBINE") == "1":
                    first_output = output[: counts[rank]].clone()
                    first_weights = (
                        out_weights[: counts[rank]].clone() if use_weights else None
                    )
                    # No host synchronization: the next push must not
                    # overwrite the peer's still-live previous inbox reads.
                    output, out_weights = op.combine(
                        expert.neg(),
                        out[1][:received].neg() if use_weights else None,
                        indices[rank],
                        routing=handle,
                    )
                    sign = -1
                reference = torch.zeros_like(source_inputs[rank], dtype=torch.float32)
                for peer in range(WS):
                    transformed = (
                        source_inputs[rank] * (peer + 1) + (peer + 1) / 7
                    ).to(dtype)
                    reference += transformed.float() * memberships[rank][:, peer, None]
                torch.testing.assert_close(
                    output[: counts[rank]],
                    (sign * reference).to(dtype),
                    rtol=0,
                    atol=0,
                )
                if first_output is not None:
                    torch.testing.assert_close(
                        first_output, reference.to(dtype), rtol=0, atol=0
                    )
                if use_weights:
                    expected_weights = torch.zeros_like(weights[rank])
                    for peer in range(WS):
                        expected_weights += (
                            weights[rank] * memberships[rank][:, peer, None]
                        )
                    torch.testing.assert_close(
                        out_weights[: counts[rank]],
                        sign * expected_weights,
                        rtol=0,
                        atol=0,
                    )
                    if first_weights is not None:
                        torch.testing.assert_close(
                            first_weights, expected_weights, rtol=0, atol=0
                        )
            torch.cuda.synchronize()
            if rank == 0:
                print(
                    f"PASS {dtype} H={hidden} K={topk} capacity={capacity} counts={counts} route={route} replay={use_handle}",
                    flush=True,
                )

        # Exercise allocation reuse between operators, after every rank has
        # finished using the old IPC mappings.
        torch.distributed.barrier()
        del op
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.barrier()


def worker(rank):
    try:
        run_worker(rank)
    except Exception:
        traceback.print_exc()
        raise


def main():
    required_env = {
        "MORI_RDNA4_EP": "1",
        "MORI_EP_COMM": "shmem",
        "MORI_SHMEM_MODE": "static_heap",
    }
    for name, value in required_env.items():
        if os.environ.get(name) != value:
            raise RuntimeError(f"set {name}={value} before running this check")
    if torch.cuda.device_count() < WS:
        raise RuntimeError(f"select {WS} gfx1201 GPUs with HIP_VISIBLE_DEVICES")
    for device in range(WS):
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch != "gfx1201":
            raise RuntimeError(
                f"device {device} is {arch}; this check requires gfx1201"
            )
    manager = start_torch_dist_process_manager(world_size=WS)
    try:
        for _ in range(WS):
            manager.task_queue.put((worker, []))
        assert_worker_results(manager, WS)
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
