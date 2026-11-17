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
"""AsyncLL segmented receive end to end through aiter fused_moe (EP8, sdma).

    torchrun --standalone --nproc_per_node=8 test_asyncll_aiter.py

Skips (exit 0, "# SKIP") when aiter is not importable. One op built with
async_ll_recv_layout="segmented" runs every round twice on the same inputs: a
compact round (the consumer as integrations call it today) and a segmented
round consumed the way async_ll_recv_layout documents:

    fused_moe(recv_x, w1, w2, recv_w, recv_idx, expert_mask,
              num_local_tokens=total_recv,          # world_size * S here
              output=op.combine_in_view())          # or any tensor (copied)

Both combined outputs must match each other and an fp32 reference of the whole
MoE computed on the source rank. Dtypes: bf16 (QuantType.No) and fp8 block
scale (QuantType.per_128x128, fp32 scales per 128 elements, the configuration
whose 1-stage path cannot take num_local_tokens=None). The per-rank token counts
are uneven, one rank sends nothing, and the strides cover exact, padded and full
capacity. Every rank's segment error counters must stay at zero.
"""

import os
import sys

import torch
import torch.distributed as dist
from mori.cco.communicator import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig, EpDispatchCombineOp

try:
    import aiter
    from aiter import ActivationType, QuantType, pertoken_quant
    from aiter.fused_moe import fused_moe
    from aiter.ops.shuffle import shuffle_weight
except Exception as exc:  # noqa: BLE001 -- any import failure means "no aiter"
    aiter = None
    _AITER_ERROR = exc

HIDDEN, INTER, EPR, TOPK, CAP = 2048, 256, 8, 6, 512
BLK = 128


def _rank_counts(world, base):
    counts = [max(0, base - 37 * r + (r % 3) * 11) for r in range(world)]
    counts[world // 2] = 0  # a rank with nothing to send still receives
    return [min(CAP, c) for c in counts]


def _block_quant(w):
    """Per (BLK, BLK) block fp8 quantization of [E, N, K], aiter's layout."""
    e, n, k = w.shape
    tmp = (
        w.view(e, n // BLK, BLK, k // BLK, BLK)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, n // BLK, k // BLK, BLK * BLK)
        .contiguous()
    )
    q, scale = pertoken_quant(tmp, quant_dtype=torch.float8_e4m3fn)
    q = (
        q.view(e, n // BLK, k // BLK, BLK, BLK)
        .permute(0, 1, 3, 2, 4)
        .reshape(e, n, k)
        .contiguous()
    )
    return q, scale.view(e, -1)


def _reference(x, w1, w2, weights, idx):
    """fp32 MoE over every expert: SiLU(gate) * up, gate the first INTER rows."""
    out = torch.zeros(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    xf = x.float()
    for k in range(idx.shape[1]):
        for e in idx[:, k].unique().tolist():
            rows = (idx[:, k] == e).nonzero().squeeze(1)
            h = xf[rows] @ w1[e].float().t()
            act = torch.nn.functional.silu(h[:, :INTER]) * h[:, INTER:]
            out[rows] += weights[rows, k, None] * (act @ w2[e].float().t())
    return out


def _scales_f32(s, rows):
    """Recv scales (packed int32 words) as [rows, HIDDEN // BLK] float32."""
    return s[:rows].contiguous().view(torch.float32)


def run(rank, world, comm, fp8, dev):
    gen = torch.Generator(device=dev).manual_seed(1234)  # same weights everywhere
    n_exp = EPR * world
    w1 = torch.randn(n_exp, 2 * INTER, HIDDEN, generator=gen, device=dev) / 16
    w2 = torch.randn(n_exp, HIDDEN, INTER, generator=gen, device=dev) / 16
    w1, w2 = w1.to(torch.bfloat16), w2.to(torch.bfloat16)
    local = slice(rank * EPR, (rank + 1) * EPR)
    mask = torch.zeros(n_exp, dtype=torch.int32, device=dev)
    mask[local] = 1
    kw = dict(activation=ActivationType.Silu, expert_mask=mask)
    if fp8:
        w1q, w1s = _block_quant(w1[local].contiguous())
        w2q, w2s = _block_quant(w2[local].contiguous())
        moe_w1, moe_w2 = shuffle_weight(w1q, (16, 16)), shuffle_weight(w2q, (16, 16))
        kw.update(quant_type=QuantType.per_128x128, w1_scale=w1s, w2_scale=w2s)
        extra = dict(
            dispatch_data_type=torch.float8_e4m3fn,
            combine_data_type=torch.bfloat16,
            scale_dim=HIDDEN // BLK,
            scale_type_size=4,
        )
    else:
        moe_w1 = shuffle_weight(w1[local].contiguous(), (16, 16))
        moe_w2 = shuffle_weight(w2[local].contiguous(), (16, 16))
        kw.update(quant_type=QuantType.No)
        extra = {}
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=HIDDEN,
        max_num_inp_token_per_rank=CAP,
        num_experts_per_rank=EPR,
        num_experts_per_token=TOPK,
        data_type=torch.bfloat16,
        kernel_backend="hip",
        async_ll=True,
        async_ll_transport="sdma",
        async_ll_recv_layout="segmented",
        **extra,
    )
    tag0 = "fp8_blockscale" if fp8 else "bf16"
    failures = 0
    with EpDispatchCombineOp(cfg, comm) as op:
        # (token count base, stride beyond the largest count, output into arena)
        for r, (base, pad, in_place) in enumerate(
            ((300, 0, True), (CAP, 0, True), (128, 64, False), (64, None, True))
        ):
            counts = _rank_counts(world, base)
            n = counts[rank]
            g = torch.Generator(device=dev).manual_seed(77 + 1000 * r + rank)
            x = torch.randn(n, HIDDEN, generator=g, device=dev).to(torch.bfloat16)
            idx = (
                torch.stack(
                    [
                        torch.randperm(n_exp, generator=g, device=dev)[:TOPK]
                        for _ in range(n)
                    ]
                ).to(torch.int32)
                if n
                else torch.empty(0, TOPK, dtype=torch.int32, device=dev)
            )
            w = torch.rand(n, TOPK, generator=g, device=dev)
            scales = None
            if fp8 and n == 0:
                x_in = x.to(torch.float8_e4m3fn)
                scales = torch.zeros(0, HIDDEN // BLK, device=dev)
                x_ref = x.float()
            elif fp8:
                q, s = pertoken_quant(
                    x.view(-1, HIDDEN // BLK, BLK), quant_dtype=torch.float8_e4m3fn
                )
                x_in, scales = q.view(n, HIDDEN), s.view(n, HIDDEN // BLK).contiguous()
                x_ref = (q.float() * s).view(n, HIDDEN)
            else:
                x_in, x_ref = x, x.float()
            ref = _reference(x_ref, w1, w2, w, idx)
            stride = CAP if pad is None else min(CAP, max(max(counts), 1) + pad)

            results = {}
            for layout in ("compact", "segmented"):
                seg = layout == "segmented"
                out, out_w, out_s, out_idx, total, routing = op.dispatch(
                    x_in,
                    w,
                    scales,
                    idx,
                    return_routing=True,
                    segment_stride=stride if seg else None,
                )
                rows = world * stride if seg else out.shape[0]
                moe_kw = dict(kw)
                if fp8:
                    moe_kw["a1_scale"] = _scales_f32(out_s, rows)
                dst = None
                if seg and in_place:
                    dst = op.combine_in_view()
                elif seg:
                    dst = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
                y = fused_moe(
                    out[:rows],
                    moe_w1,
                    moe_w2,
                    out_w[:rows],
                    out_idx[:rows],
                    num_local_tokens=total,
                    output=dst,
                    dtype=torch.bfloat16,
                    **moe_kw,
                )
                combined, _ = op.combine(y, routing=routing)
                torch.cuda.synchronize()
                results[layout] = combined[:n].float().clone()
                if seg and int(total[0].item()) != rows:
                    print(
                        f"[rank {rank}] {tag0} r{r}: total_recv "
                        f"{int(total[0].item())} != rows {rows}",
                        flush=True,
                    )
                    failures += 1
            dist.barrier()

            tag = f"{tag0} round={r} stride={stride} in_place={in_place}"
            scale = ref.abs().max().clamp_min(1e-3) if n else None
            tol = 0.08 if fp8 else 0.03
            for layout, got in results.items():
                err = ((got - ref).abs().max() / scale).item() if n else 0.0
                if err > tol:
                    print(
                        f"[rank {rank}] {tag}: {layout} vs reference rel err "
                        f"{err:.4f} > {tol}",
                        flush=True,
                    )
                    failures += 1
            if n:
                diff = (
                    (results["compact"] - results["segmented"]).abs().max() / scale
                ).item()
                if diff > 1e-2:
                    print(
                        f"[rank {rank}] {tag}: segmented vs compact rel diff "
                        f"{diff:.4f}",
                        flush=True,
                    )
                    failures += 1
        errors = op.segment_error_counts()
        if errors != (0, 0):
            print(f"[rank {rank}] {tag0}: segment error counts {errors}", flush=True)
            failures += 1
    return failures


def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if aiter is None:
        if rank == 0:
            print(f"# SKIP: aiter not importable ({_AITER_ERROR!r})", flush=True)
        return 0
    dist.init_process_group(backend="gloo")
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    failures = 0
    for fp8 in (False, True):
        objs = [Communicator.get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(objs, src=0)
        win = world * HIDDEN * CAP * 2 * 4 + (1 << 24)
        with Communicator.init(
            world, rank, objs[0], per_rank_vmm=6 * win + (1 << 28)
        ) as comm:
            failures += run(rank, world, comm, fp8, dev)
        if rank == 0:
            print(
                f"# asyncll aiter {'fp8 blockscale' if fp8 else 'bf16'}: done",
                flush=True,
            )
    t = torch.tensor([failures], dtype=torch.int64)
    dist.all_reduce(t)
    if rank == 0:
        print(
            f"# {'PASS' if t.item() == 0 else 'FAIL'}: {t.item()} failed checks "
            f"across {world} ranks, bf16 and fp8 blockscale through aiter fused_moe",
            flush=True,
        )
    dist.destroy_process_group()
    return 0 if t.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
