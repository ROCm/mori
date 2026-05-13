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
import pytest
from tests.python.utils import TorchDistProcessManager, data_type_supported
import mori
import torch
import torch.distributed as dist

TORCH_FLOAT4_E2M1FN_X2 = getattr(torch, "float4_e2m1fn_x2", None)

# OCP/FNUZ FP8 e4m3 max finite. Matches kCombineInternalFp8MaxFinite on the
# device side. Used as the threshold for blockwise scale-active detection.
FP8_E4M3_MAX_FINITE = 448.0

INPUT_DIST_CHOICES = ("normal", "uniform", "lognormal", "two_bucket")


def _is_fp4x2_dtype(dtype):
    return TORCH_FLOAT4_E2M1FN_X2 is not None and dtype is TORCH_FLOAT4_E2M1FN_X2


def _generate_input_tensor(
    shape,
    *,
    dtype,
    device,
    generator,
    input_dist="normal",
    input_scale=1.0,
    input_shift=0.0,
    force_scale_active=False,
    block_elems=None,
    fp8_max=FP8_E4M3_MAX_FINITE,
):
    """Generate a [N, H] input tensor with the requested distribution.

    `input_scale` / `input_shift` are applied uniformly after sampling.
    When `force_scale_active=True`, each scale block (last dim partitioned
    into chunks of `block_elems`) is guaranteed to contain at least one
    element with abs > `fp8_max`, so the kernel always exercises the
    scale-active path.
    """
    if input_dist not in INPUT_DIST_CHOICES:
        raise ValueError(
            f"Unsupported input_dist={input_dist!r}; "
            f"choose from {INPUT_DIST_CHOICES}"
        )

    n, h = shape

    if input_dist == "normal":
        x = torch.randn(n, h, dtype=torch.float32, device=device, generator=generator)
    elif input_dist == "uniform":
        # Uniform on [-1, 1] before scale/shift.
        u = torch.rand(n, h, dtype=torch.float32, device=device, generator=generator)
        x = u * 2.0 - 1.0
    elif input_dist == "lognormal":
        # Centered lognormal: exp(N(0,1)) - 1 produces a long right tail.
        z = torch.randn(n, h, dtype=torch.float32, device=device, generator=generator)
        x = torch.exp(z) - 1.0
    elif input_dist == "two_bucket":
        # Mostly N(0,1), 1% drawn from N(0, 16) to model long-tail activations.
        # The heavy bucket scale is 16x the base, multiplied by `input_scale`
        # to allow further widening from the CLI.
        base = torch.randn(
            n, h, dtype=torch.float32, device=device, generator=generator
        )
        heavy = (
            torch.randn(n, h, dtype=torch.float32, device=device, generator=generator)
            * 16.0
        )
        mask = (
            torch.rand(n, h, dtype=torch.float32, device=device, generator=generator)
            < 0.01
        )
        x = torch.where(mask, heavy, base)
    else:
        # Defensive; INPUT_DIST_CHOICES should already cover this.
        raise ValueError(f"Unsupported input_dist={input_dist!r}")

    if input_scale != 1.0:
        x.mul_(float(input_scale))
    if input_shift != 0.0:
        x.add_(float(input_shift))

    if force_scale_active:
        if block_elems is None or block_elems <= 0:
            raise ValueError("force_scale_active=True requires a positive block_elems")
        # Ensure every (token, scale-block) has |x| > fp8_max for at least
        # one element. We overwrite the first lane of each block with a
        # sign-alternating value of magnitude 2*fp8_max so that the kernel
        # max-reduce always sees a value above the threshold.
        if n > 0 and h > 0:
            num_blocks = (h + block_elems - 1) // block_elems
            sentinel = float(fp8_max) * 2.0
            for b in range(num_blocks):
                col = b * block_elems
                if col >= h:
                    break
                # Alternate signs across tokens so we don't bias the
                # accumulate output direction.
                signs = torch.where(
                    torch.arange(n, device=device) % 2 == 0,
                    torch.tensor(sentinel, device=device),
                    torch.tensor(-sentinel, device=device),
                )
                x[:, col] = signs

    if _is_fp4x2_dtype(dtype):
        # FP4 path is not driven by this helper; callers should not pass
        # FP4 dtype through here.
        raise ValueError("_generate_input_tensor does not support FP4x2 dtype")

    return x.to(dtype)


def compute_scale_active_stats(
    all_rank_input,
    *,
    scale_dim,
    fp8_max=FP8_E4M3_MAX_FINITE,
):
    """Compute scale-active statistics matching the device-side blockwise quant.

    For a token with hidden dimension H and `scale_dim` scale blocks, the
    block size in elements is `block_elems = ceil(H / scale_dim)` (the
    same partition the device kernel uses). A block is considered
    "scale-active" when its element-wise max absolute value exceeds
    `fp8_max`. A token is considered "any-scaled" when at least one of
    its blocks is scale-active.

    Returns a list of per-rank stat dicts (None entries for empty ranks)
    plus an aggregated dict over all non-empty ranks.
    """
    per_rank = []
    block_scaled_counts = []
    block_total_counts = []
    token_any_scaled_counts = []
    token_total_counts = []
    all_token_max_abs = []

    for r, x in enumerate(all_rank_input):
        if x is None or x.numel() == 0:
            per_rank.append(None)
            continue

        n, h = x.shape
        x_f = x.to(torch.float32).abs()
        block_elems = (h + scale_dim - 1) // scale_dim
        padded_h = block_elems * scale_dim
        if padded_h != h:
            pad = torch.zeros(n, padded_h - h, device=x_f.device, dtype=x_f.dtype)
            x_f = torch.cat([x_f, pad], dim=1)

        blocks = x_f.view(n, scale_dim, block_elems)
        block_max = blocks.amax(dim=2)
        block_scaled = block_max > fp8_max
        token_any_scaled = block_scaled.any(dim=1)
        token_max_abs = block_max.amax(dim=1)

        token_max_abs_cpu = token_max_abs.detach().to("cpu").float()
        per_rank.append(
            {
                "rank": r,
                "num_tokens": int(n),
                "block_elems": int(block_elems),
                "scale_dim": int(scale_dim),
                "token_any_scaled_ratio": float(token_any_scaled.float().mean()),
                "block_scaled_ratio": float(block_scaled.float().mean()),
                "max_abs_p50": float(torch.quantile(token_max_abs_cpu, 0.50)),
                "max_abs_p90": float(torch.quantile(token_max_abs_cpu, 0.90)),
                "max_abs_p99": float(torch.quantile(token_max_abs_cpu, 0.99)),
                "max_abs_max": float(token_max_abs_cpu.max()),
            }
        )

        block_scaled_counts.append(int(block_scaled.sum().item()))
        block_total_counts.append(int(block_scaled.numel()))
        token_any_scaled_counts.append(int(token_any_scaled.sum().item()))
        token_total_counts.append(int(token_any_scaled.numel()))
        all_token_max_abs.append(token_max_abs_cpu)

    if all_token_max_abs:
        merged = torch.cat(all_token_max_abs)
        block_total = sum(block_total_counts) or 1
        token_total = sum(token_total_counts) or 1
        aggregate = {
            "num_ranks": len(all_token_max_abs),
            "num_tokens_total": token_total,
            "block_scaled_ratio": sum(block_scaled_counts) / block_total,
            "token_any_scaled_ratio": sum(token_any_scaled_counts) / token_total,
            "max_abs_p50": float(torch.quantile(merged, 0.50)),
            "max_abs_p90": float(torch.quantile(merged, 0.90)),
            "max_abs_p99": float(torch.quantile(merged, 0.99)),
            "max_abs_max": float(merged.max()),
        }
    else:
        aggregate = None

    return per_rank, aggregate


def format_scale_stats_report(
    per_rank, aggregate, *, scale_dim, fp8_max=FP8_E4M3_MAX_FINITE
):
    """Format scale-active stats as a multi-line human-readable report."""
    lines = []
    lines.append(
        f"[scale-stats] scale_dim={scale_dim}, fp8_max={fp8_max:g} "
        f"(token_any_scaled_ratio = fraction of tokens with at least one block "
        f"whose max|x| > fp8_max; block_scaled_ratio = fraction of all blocks "
        f"with max|x| > fp8_max)"
    )
    for stat in per_rank:
        if stat is None:
            continue
        lines.append(
            f"  rank {stat['rank']:>2d}: ntok={stat['num_tokens']:>5d} "
            f"block_elems={stat['block_elems']:>3d} "
            f"any_scaled={stat['token_any_scaled_ratio']:.4f} "
            f"block_scaled={stat['block_scaled_ratio']:.4f} "
            f"max|x| p50={stat['max_abs_p50']:.3f} "
            f"p90={stat['max_abs_p90']:.3f} "
            f"p99={stat['max_abs_p99']:.3f} "
            f"max={stat['max_abs_max']:.3f}"
        )
    if aggregate is not None:
        lines.append(
            f"  AGG  : ntok={aggregate['num_tokens_total']:>5d} "
            f"any_scaled={aggregate['token_any_scaled_ratio']:.4f} "
            f"block_scaled={aggregate['block_scaled_ratio']:.4f} "
            f"max|x| p50={aggregate['max_abs_p50']:.3f} "
            f"p90={aggregate['max_abs_p90']:.3f} "
            f"p99={aggregate['max_abs_p99']:.3f} "
            f"max={aggregate['max_abs_max']:.3f}"
        )
    return "\n".join(lines)


_FP4_E2M1_LUT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def unpack_fp4x2(fp4x2_tensor, dtype=torch.bfloat16):
    """Unpack float4_e2m1fn_x2 tensor [*, H] to float [*, H*2].

    Each fp4x2 element (1 byte) stores two FP4 E2M1 values:
    low nibble = first value, high nibble = second value.
    """
    raw = fp4x2_tensor.view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=dtype, device=raw.device)
    result = torch.stack([lut[low.long()], lut[high.long()]], dim=-1)
    return result.reshape(*raw.shape[:-1], raw.shape[-1] * 2)


def _all_data_types():
    """Return parametrize list of all supported data types with skipif marks."""
    types = [
        torch.bfloat16,
        pytest.param(
            torch.float8_e4m3fnuz,
            marks=pytest.mark.skipif(
                not data_type_supported(torch.float8_e4m3fnuz),
                reason="Skip float8_e4m3fnuz, it is not supported",
            ),
        ),
        pytest.param(
            torch.float8_e4m3fn,
            marks=pytest.mark.skipif(
                not data_type_supported(torch.float8_e4m3fn),
                reason="Skip float8_e4m3fn, it is not supported",
            ),
        ),
    ]
    if TORCH_FLOAT4_E2M1FN_X2 is not None:
        types.append(
            pytest.param(
                TORCH_FLOAT4_E2M1FN_X2,
                marks=pytest.mark.skipif(
                    not data_type_supported(TORCH_FLOAT4_E2M1FN_X2),
                    reason="Skip float4_e2m1fn_x2, it is not supported",
                ),
            )
        )
    return types


def start_torch_dist_process_manager(world_size=8, disable_p2p=False):
    if disable_p2p:
        torch.cuda.empty_cache()
        import os

        os.environ["MORI_DISABLE_P2P"] = "1"

    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
        print("Multiprocessing start method set to spawn")
    except RuntimeError:
        pass

    manager = TorchDistProcessManager()
    manager.start_workers(world_size=world_size)
    return manager


def assert_worker_results(manager, world_size):
    results = []
    for _ in range(world_size):
        rank, result = manager.result_queue.get()
        results.append((rank, result))

    for _, result in sorted(results, key=lambda item: item[0]):
        if result is not None:
            pytest.assume(False, result)


class EpDispatchCombineTestCase:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda", self.config.rank)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(123)

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier()

    def gen_test_data(
        self,
        use_max_token_num=False,
        routing="random",
        num_token_override=None,
        input_dist="normal",
        input_scale=1.0,
        input_shift=0.0,
        force_scale_active=False,
        combine_scale_dim=None,
    ):
        """Generate test data."""
        if num_token_override is not None:
            assert len(num_token_override) == self.config.world_size
            assert min(num_token_override) >= 0
            assert max(num_token_override) <= self.config.max_num_inp_token_per_rank
            num_token = torch.tensor(num_token_override, device=self.device)
        elif use_max_token_num:
            num_token = torch.tensor(
                [
                    self.config.max_num_inp_token_per_rank
                    for _ in range(self.config.world_size)
                ]
            ).to(self.device)
        else:
            num_token = torch.randint(
                0,
                self.config.max_num_inp_token_per_rank + 1,
                [self.config.world_size],
                generator=self.rng,
                device=self.device,
            )

        total_experts = self.config.num_experts_per_rank * self.config.world_size

        all_rank_indices = []
        for r in range(self.config.world_size):
            n = int(num_token[r])
            if routing == "round_robin":
                indices = torch.empty(
                    n, self.config.num_experts_per_token, dtype=torch.int64
                )
                for i in range(n):
                    base = (
                        r * self.config.max_num_inp_token_per_rank + i
                    ) * self.config.num_experts_per_token
                    for j in range(self.config.num_experts_per_token):
                        indices[i, j] = (base + j) % total_experts
            elif routing == "spread":
                # Sends exactly one expert to every rank (requires num_experts_per_token ==
                # world_size). After per-rank deduplication each rank receives every source
                # token exactly once, so total recv = max_num_inp_token_per_rank * world_size
                # (the true worst case).
                assert (
                    self.config.num_experts_per_token == self.config.world_size
                ), "spread routing requires num_experts_per_token == world_size"
                indices = torch.empty(
                    n, self.config.num_experts_per_token, dtype=torch.int64
                )
                for i in range(n):
                    for j in range(self.config.num_experts_per_token):
                        indices[i, j] = j * self.config.num_experts_per_rank
            elif routing == "all_to_one":
                indices = torch.zeros(
                    n, self.config.num_experts_per_token, dtype=torch.int64
                )
            else:
                indices = torch.empty(
                    n, self.config.num_experts_per_token, dtype=torch.int64
                )
                for i in range(n):
                    perm = torch.randperm(
                        total_experts,
                        generator=self.rng,
                        device=self.device,
                    )
                    indices[i] = perm[: self.config.num_experts_per_token]
            all_rank_indices.append(indices.to(torch.int32).to(self.device))

        all_rank_weights = [
            torch.rand(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            for r in range(self.config.world_size)
        ]

        all_rank_scales = [
            torch.rand(
                num_token[r],
                self.config.scale_dim,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            for r in range(self.config.world_size)
        ]
        if self.config.scale_type_size == 1:
            all_rank_scales = [t.to(torch.float8_e4m3fnuz) for t in all_rank_scales]

        # Pre-compute the device-side block partition so force_scale_active
        # writes a sentinel into the same block layout the kernel uses.
        # For fp8_blockwise the caller should pass `combine_scale_dim` from
        # op._fp8_blockwise_combine_scale_dim; otherwise we fall back to the
        # user-config scale_dim (FP4 / non-blockwise paths).
        partition_scale_dim = (
            combine_scale_dim if combine_scale_dim else self.config.scale_dim
        )
        block_elems = (
            (self.config.hidden_dim + partition_scale_dim - 1) // partition_scale_dim
            if partition_scale_dim > 0
            else self.config.hidden_dim
        )

        all_rank_input = []
        for r in range(self.config.world_size):
            n_r = int(num_token[r])
            if _is_fp4x2_dtype(self.config.data_type):
                fp4_bytes = torch.randint(
                    0,
                    256,
                    (n_r, self.config.hidden_dim),
                    dtype=torch.uint8,
                    generator=self.rng,
                    device=self.device,
                )
                all_rank_input.append(fp4_bytes.view(torch.float4_e2m1fn_x2))
            else:
                all_rank_input.append(
                    _generate_input_tensor(
                        (n_r, self.config.hidden_dim),
                        dtype=self.config.data_type,
                        device=self.device,
                        generator=self.rng,
                        input_dist=input_dist,
                        input_scale=input_scale,
                        input_shift=input_shift,
                        force_scale_active=force_scale_active,
                        block_elems=block_elems,
                    )
                )

        return (
            num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        )

    def check_dispatch_result(
        self,
        op,
        test_data,
        dispatch_output,
        dispatch_weights,
        dispatch_scales,
        dispatch_indices,
        dispatch_recv_num_token,
    ):
        self.sync()
        (
            _,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        src_token_pos = op.get_dispatch_src_token_pos()

        for i, pos in enumerate(src_token_pos):
            src_rank, src_id = op.decode_send_flat_idx(pos)
            if _is_fp4x2_dtype(self.config.data_type):
                assert torch.equal(
                    all_rank_input[src_rank][src_id].view(torch.uint8),
                    dispatch_output[i].view(torch.uint8),
                )
            else:
                assert torch.equal(all_rank_input[src_rank][src_id], dispatch_output[i])
            if dispatch_weights is not None:
                assert torch.equal(
                    all_rank_weights[src_rank][src_id], dispatch_weights[i]
                )
            if dispatch_scales is not None:
                assert torch.equal(
                    all_rank_scales[src_rank][src_id], dispatch_scales[i]
                )
            assert torch.equal(all_rank_indices[src_rank][src_id], dispatch_indices[i])
        assert len(torch.unique(src_token_pos)) == len(src_token_pos)
        assert len(src_token_pos) == dispatch_recv_num_token[0]

    def check_combine_result(
        self,
        op,
        test_data,
        combine_output,
        combine_output_weight=None,
        combine_data_type=None,
    ):
        self.sync()
        all_rank_num_token = test_data[0]
        all_rank_indices = test_data[1]
        all_rank_input = test_data[2]
        all_rank_weights = test_data[3]

        if combine_data_type is None:
            combine_data_type = self.config.data_type

        if _is_fp4x2_dtype(combine_data_type):
            return

        for i in range(all_rank_num_token[self.config.rank]):
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in all_rank_indices[self.config.rank][i].cpu().tolist()
            ]
            unique_pes = len(set(pes))

            inp = all_rank_input[self.config.rank][i]
            if _is_fp4x2_dtype(self.config.data_type):
                inp_converted = unpack_fp4x2(
                    inp.unsqueeze(0), dtype=combine_data_type
                ).squeeze(0)
            else:
                inp_converted = inp.to(combine_data_type)

            got, expected = combine_output[i], (
                inp_converted.to(torch.float32) * unique_pes
            ).to(combine_data_type)

            atol, rtol = 1e-2, 1e-2
            if getattr(self.config, "quant_type", "none") == "fp8_direct_cast":
                atol, rtol = 1e-1, 1e-1
            elif getattr(self.config, "quant_type", "none") == "fp8_blockwise":
                # FP8 E4M3 quantization can exceed 5% after combine accumulation.
                atol, rtol = 7e-2, 7e-2
            result_match = torch.allclose(
                got.float(), expected.float(), atol=atol, rtol=rtol
            )
            if not result_match:
                diff = (got.float() - expected.float()).abs()
                tol = atol + rtol * expected.float().abs()
                max_idx = int(diff.argmax().item())
                print(f"Rank[{self.config.rank}] result mismatch for token {i}:")
                print(
                    f"Rank[{self.config.rank}]   indices[{i}]: {all_rank_indices[self.config.rank][i].cpu().tolist()}"
                )
                print(f"Rank[{self.config.rank}]   pes: {pes}")
                print(f"Rank[{self.config.rank}]   unique_pes: {unique_pes}")
                print(
                    f"Rank[{self.config.rank}]   max diff: idx={max_idx}, "
                    f"diff={float(diff[max_idx])}, tol={float(tol[max_idx])}"
                )
                print(f"Rank[{self.config.rank}]   got: {got}")
                print(f"Rank[{self.config.rank}]   expected : {expected}")
                print(
                    f"Rank[{self.config.rank}]   input : {all_rank_input[self.config.rank][i].to(torch.float32)}"
                )
            assert result_match

            if combine_output_weight is not None:
                got_weight, expected_weight = (
                    combine_output_weight[i],
                    all_rank_weights[self.config.rank][i] * unique_pes,
                )
                weight_match = torch.allclose(
                    got_weight, expected_weight, atol=1e-5, rtol=1e-5
                )
                if not weight_match:
                    print(f"Rank[{self.config.rank}] Weight mismatch for token {i}:")
                    print(
                        f"Rank[{self.config.rank}]   indices[{i}]: {all_rank_indices[self.config.rank][i].cpu().tolist()}"
                    )
                    print(f"Rank[{self.config.rank}]   pes: {pes}")
                    print(f"Rank[{self.config.rank}]   unique_pes: {unique_pes}")
                    print(f"Rank[{self.config.rank}]   got_weight: {got_weight}")
                    print(
                        f"Rank[{self.config.rank}]   expected_weight (weights[{i}] * {unique_pes}): {expected_weight}"
                    )
                assert weight_match

    def run_test_once(self, op, test_data, check_results=True):
        (
            _,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indices,
            dispatch_recv_num_token,
        ) = op.dispatch(
            all_rank_input[self.config.rank],
            all_rank_weights[self.config.rank],
            all_rank_scales[self.config.rank],
            all_rank_indices[self.config.rank],
        )
        self.sync()
        if check_results:
            self.check_dispatch_result(
                op,
                test_data,
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            )

        total_recv_num_token = dispatch_recv_num_token[0].item()
        if not self.config.use_external_inp_buf:
            combine_input = op.get_registered_combine_input_buffer(
                self.config.data_type
            )
            combine_input[:total_recv_num_token, :].copy_(
                dispatch_output[:total_recv_num_token, :]
            )
        combine_output, combine_output_weight = op.combine(
            dispatch_output, dispatch_weights, dispatch_indices, call_reset=False
        )
        self.sync()
        if check_results:
            self.check_combine_result(
                op, test_data, combine_output, combine_output_weight
            )


def check_local_expert_count(op, dispatch_indices, dispatch_recv_num_token):
    """Verify op.local_expert_count matches a CPU reference computed from dispatch outputs.

    For each of the ``total_recv`` received tokens, any expert index ``e`` that
    maps to the local rank (``e // num_experts_per_rank == rank``) contributes
    one count to ``local_expert_count[e % num_experts_per_rank]``.
    """
    rank = op.config.rank
    num_experts_per_rank = op.config.num_experts_per_rank
    total_recv = dispatch_recv_num_token[0].item()
    received_indices = dispatch_indices[:total_recv].cpu()
    expected = torch.zeros(num_experts_per_rank, dtype=torch.int32)
    for tok_idx in range(total_recv):
        for e in received_indices[tok_idx].tolist():
            if e // num_experts_per_rank == rank:
                expected[e % num_experts_per_rank] += 1
    actual = op.local_expert_count.cpu()
    assert torch.equal(actual, expected), (
        f"Rank {rank}: local_expert_count mismatch:\n"
        f"  got:      {actual.tolist()}\n"
        f"  expected: {expected.tolist()}"
    )


def run_ep_dispatch_local_expert_count_test(config):
    """Run a dispatch + local_expert_count test for any kernel type.

    Handles both single-call kernels (IntraNode, InterNodeV1/LL) and the
    split-phase AsyncLL kernel (dispatch_send + dispatch_recv).  After the
    kernel completes, verifies ``op.local_expert_count`` against a CPU
    reference computed from the received indices.
    """
    op = mori.ops.EpDispatchCombineOp(config)
    test_case = EpDispatchCombineTestCase(config)
    test_data = test_case.gen_test_data()

    (_, all_rank_indices, all_rank_input, all_rank_weights, all_rank_scales) = test_data
    rank = config.rank

    kt = config.kernel_type
    asyncll_type = mori.ops.EpDispatchCombineKernelType.AsyncLL
    if kt.value == asyncll_type.value:
        # AsyncLL: recv kernels fill the output indices, so local_expert_count
        # must be requested on dispatch_recv (not dispatch_send).
        (_, _, _, dispatch_indices, dispatch_recv_num_token) = op.dispatch_send(
            all_rank_input[rank],
            all_rank_weights[rank],
            all_rank_scales[rank],
            all_rank_indices[rank],
        )
        op.dispatch_recv(call_local_expert_count=True)
    else:
        (_, _, _, dispatch_indices, dispatch_recv_num_token) = op.dispatch(
            all_rank_input[rank],
            all_rank_weights[rank],
            all_rank_scales[rank],
            all_rank_indices[rank],
            call_local_expert_count=True,
        )

    torch.cuda.synchronize()
    check_local_expert_count(op, dispatch_indices, dispatch_recv_num_token)


def run_ep_dispatch_combine_test(
    config,
    test_case_cls,
    use_max_token_num=False,
    routing=None,
    num_token_override=None,
    check_results=True,
):
    op = mori.ops.EpDispatchCombineOp(config)
    test_case = test_case_cls(config)
    gen_kwargs = {}
    if use_max_token_num:
        gen_kwargs["use_max_token_num"] = True
    if routing is not None:
        gen_kwargs["routing"] = routing
    if num_token_override is not None:
        gen_kwargs["num_token_override"] = num_token_override
    test_data = test_case.gen_test_data(**gen_kwargs)
    test_case.run_test_once(op, test_data, check_results=check_results)
