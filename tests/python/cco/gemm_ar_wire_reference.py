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
"""Check FP8 scatter and collective semantics separately from quantization error."""

import torch
import torch.distributed as dist
from mori.tensor_utils import from_gpu_ptr


def check_wire(cfg, pointer, baseline_partial, output, rank, world):
    rows, n = cfg.m, cfg.n
    raw = from_gpu_ptr(pointer + cfg.input_off, (rows, n), torch.float8_e4m3fn)
    scales = from_gpu_ptr(
        pointer + cfg.input_scale_off, (rows, n // 256), torch.float32
    )
    source = baseline_partial.float().reshape(rows, n // 256, 256)
    expected_scale = source.abs().amax(-1) / 448.0
    expected_scale = torch.where(expected_scale == 0, 1.0, expected_scale)
    expected_q = (source / expected_scale[..., None]).to(torch.float8_e4m3fn).float()
    expected = (expected_q * expected_scale[..., None]).reshape(rows, n)
    actual = (raw.float().reshape(rows, n // 256, 256) * scales[..., None]).reshape(
        rows, n
    )
    if not bool(torch.isfinite(actual).all()):
        raise RuntimeError("nonfinite FP8 scatter payload/scales")
    quant_rel = ((actual - expected).norm() / expected.norm()).item()
    actual_q = raw.float().reshape_as(expected_q)
    different = actual_q != expected_q
    midpoint = (actual_q + expected_q) * 0.5
    scaled = source / expected_scale[..., None]
    near_midpoint = (scaled - midpoint).abs() <= 4e-6 * scaled.abs() + 1e-7
    expected_bytes = (
        expected_q.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)
    )
    actual_bytes = raw.view(torch.uint8).reshape_as(expected_q).to(torch.int16)
    adjacent = (expected_bytes - actual_bytes).abs() <= 1
    scale_rel = ((scales - expected_scale).norm() / expected_scale.norm()).item()
    if bool((different & ~(near_midpoint & adjacent)).any()) or not scale_rel < 1e-6:
        raise RuntimeError(
            (
                "FP8 tile quantization",
                rank,
                quant_rel,
                scale_rel,
                int(torch.count_nonzero(different & ~(near_midpoint & adjacent))),
            )
        )
    local = actual.cpu()
    pieces = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(pieces, local)
    reference = torch.empty(rows, n, dtype=torch.bfloat16)
    for owner in range(world):
        lo, hi = owner * cfg.slice_rows, (owner + 1) * cfg.slice_rows
        total = pieces[owner][lo:hi].clone()
        for j in range(1, world):
            total += pieces[(owner + j) % world][lo:hi]
        reference[lo:hi] = total.to(torch.bfloat16)
    reference = reference.cuda()
    gather_stats = {}
    if cfg.fp8_gather:
        lo, hi = rank * cfg.slice_rows, (rank + 1) * cfg.slice_rows
        reduced = output[lo:hi].float()
        reduce_rel = (
            (reduced - reference[lo:hi].float()).norm()
            / reference[lo:hi].float().norm().clamp_min(1e-30)
        ).item()
        if not reduce_rel < 1e-3:
            raise RuntimeError(("scatter reduction before gather", rank, reduce_rel))
        gather_raw = from_gpu_ptr(
            pointer + cfg.gout_off + lo * n, (cfg.slice_rows, n), torch.float8_e4m3fn
        )
        gather_scale = from_gpu_ptr(
            pointer + cfg.gout_scale_slice_off(rank), (cfg.slice_rows, 1), torch.float32
        )
        expected_gs = reduced.abs().amax(-1, keepdim=True) / 448.0
        expected_gs = torch.where(expected_gs == 0, 1.0, expected_gs)
        scaled_g = reduced / expected_gs
        expected_gq = scaled_g.to(torch.float8_e4m3fn)
        actual_gq = gather_raw.float()
        different_g = actual_gq != expected_gq.float()
        midpoint_g = (actual_gq + expected_gq.float()) * 0.5
        near_g = (scaled_g - midpoint_g).abs() <= 4e-6 * scaled_g.abs() + 1e-7
        adjacent_g = (
            gather_raw.view(torch.uint8).to(torch.int16)
            - expected_gq.view(torch.uint8).to(torch.int16)
        ).abs() <= 1
        gs_rel = (
            (gather_scale - expected_gs).norm() / expected_gs.norm().clamp_min(1e-30)
        ).item()
        bad = different_g & ~(near_g & adjacent_g)
        if bool(bad.any()) or not gs_rel < 1e-6:
            raise RuntimeError(
                ("FP8 gather quantization", rank, gs_rel, int(bad.count_nonzero()))
            )
        # Reconstruct the second communication leg from the bytes/scales
        # actually transmitted, after checking its quantizer independently.
        decoded = (actual_gq * gather_scale).to(torch.bfloat16).float().cpu()
        gathered = [torch.empty_like(decoded) for _ in range(world)]
        dist.all_gather(gathered, decoded)
        reference = torch.cat(gathered).cuda()
        reference[lo:hi] = reduced
        gather_stats = dict(
            wire_reduce_rel_l2=reduce_rel,
            wire_gather_scale_rel_l2=gs_rel,
            wire_gather_midpoint_flips=int(different_g.count_nonzero()),
        )
        print("GATHER_WIRE_CHECK", rank, gather_stats, flush=True)
    got = output.float()
    reference = reference[: got.shape[0]].float()
    comm_rel = ((got - reference).norm() / reference.norm()).item()
    if not comm_rel < 1e-3:
        raise RuntimeError(("collective beyond expected quantization", rank, comm_rel))
    return dict(
        **gather_stats,
        wire_quant_rel_l2=quant_rel,
        wire_collective_rel_l2=comm_rel,
        wire_midpoint_flips=int(torch.count_nonzero(different)),
        wire_scale_rel_l2=scale_rel,
    )
