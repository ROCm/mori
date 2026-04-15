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
import mori
import os

import torch
import torch.distributed as dist
import argparse
import time
from tqdm import tqdm

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")

kernel_type_map = {
    "v0": mori.ops.EpDispatchCombineKernelType.InterNode,
    "v1": mori.ops.EpDispatchCombineKernelType.InterNodeV1,
    "v1_ll": mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
    "async_ll": mori.ops.EpDispatchCombineKernelType.AsyncLL,
}

_CLI_TO_KERNEL_TYPE_NAME = {
    "v0": "InterNode",
    "v1": "InterNodeV1",
    "v1_ll": "InterNodeV1LL",
    "async_ll": "AsyncLL",
}

_FP4_DTYPE = getattr(torch, "float4_e2m1fn_x2", None)


def _is_fp4x2_dtype(dtype):
    return _FP4_DTYPE is not None and dtype is _FP4_DTYPE


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
    """Unpack float4_e2m1fn_x2 tensor [*, H] to float [*, H*2]."""
    raw = fp4x2_tensor.view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=dtype, device=raw.device)
    result = torch.stack([lut[low.long()], lut[high.long()]], dim=-1)
    return result.reshape(*raw.shape[:-1], raw.shape[-1] * 2)


def _save_internode_tuning_result(
    config_path,
    config,
    max_num_token,
    dispatch_hidden_dim,
    combine_hidden_dim,
    combine_data_type,
    best_disp_config,
    best_disp_bw,
    best_comb_config,
    best_comb_bw,
    best_disp_all_bw=None,  # dict with rdma/xgmi/ll/lat stats or None
    best_comb_all_bw=None,
):
    from pathlib import Path
    from mori.ops.tuning_config import (
        TuningConfigManager,
        dtype_to_config_str,
        build_config_filename,
        quant_type_to_config_str,
        kernel_type_to_config_str,
        detect_gpu_model,
    )
    from mori.jit.config import detect_gpu_arch

    gpu_arch = detect_gpu_arch()
    gpu_model = detect_gpu_model()
    kernel_type_name = kernel_type_to_config_str(config.kernel_type)
    disp_dtype_str = dtype_to_config_str(config.data_type)
    comb_dtype_str = dtype_to_config_str(combine_data_type)
    qt_str = quant_type_to_config_str(config.quant_type)

    metadata = {
        "gpu_arch": gpu_arch,
        "gpu_model": gpu_model,
        "kernel_type": kernel_type_name,
        "ep_size": config.world_size,
    }

    def _stats_avg(stats, key):
        return stats[key][2] if stats else 0.0

    dispatch_entry = {
        "dtype": disp_dtype_str,
        "num_tokens": max_num_token,
        "hidden_dim": dispatch_hidden_dim,
        "block_num": best_disp_config[0],
        "rdma_block_num": best_disp_config[2],
        "warp_per_block": best_disp_config[1],
        "bandwidth_gbps": round(best_disp_bw, 2),
        "avg_rdma_bandwidth_gbps": round(_stats_avg(best_disp_all_bw, "rdma"), 2),
        "avg_xgmi_bandwidth_gbps": round(_stats_avg(best_disp_all_bw, "xgmi"), 2),
        "avg_ll_bandwidth_gbps": round(_stats_avg(best_disp_all_bw, "ll"), 2),
        "avg_latency_us": round(_stats_avg(best_disp_all_bw, "lat"), 2),
    }

    combine_entry = {
        "dtype": comb_dtype_str,
        "num_tokens": max_num_token,
        "hidden_dim": combine_hidden_dim,
        "zero_copy": False,
        "quant_type": qt_str,
        "block_num": best_comb_config[0],
        "rdma_block_num": best_comb_config[2],
        "warp_per_block": best_comb_config[1],
        "bandwidth_gbps": round(best_comb_bw, 2),
        "avg_rdma_bandwidth_gbps": round(_stats_avg(best_comb_all_bw, "rdma"), 2),
        "avg_xgmi_bandwidth_gbps": round(_stats_avg(best_comb_all_bw, "xgmi"), 2),
        "avg_ll_bandwidth_gbps": round(_stats_avg(best_comb_all_bw, "ll"), 2),
        "avg_latency_us": round(_stats_avg(best_comb_all_bw, "lat"), 2),
    }

    if config_path == "auto":
        repo_tuning_dir = (
            Path(__file__).resolve().parents[3]
            / "python"
            / "mori"
            / "ops"
            / "tuning_configs"
        )
        dispatch_path = str(
            repo_tuning_dir
            / build_config_filename(
                gpu_arch,
                kernel_type_name,
                config.world_size,
                gpu_model,
                "dispatch",
            )
        )
        combine_path = str(
            repo_tuning_dir
            / build_config_filename(
                gpu_arch,
                kernel_type_name,
                config.world_size,
                gpu_model,
                "combine",
            )
        )
    else:
        base = config_path.rsplit(".", 1)[0] if "." in config_path else config_path
        dispatch_path = f"{base}_dispatch.json"
        combine_path = f"{base}_combine.json"

    dispatch_metadata = {**metadata, "phase": "dispatch"}
    combine_metadata = {**metadata, "phase": "combine"}

    TuningConfigManager.save_tuning_result(
        dispatch_path,
        dispatch_metadata,
        dispatch_entry,
        phase="dispatch",
    )
    TuningConfigManager.save_tuning_result(
        combine_path,
        combine_metadata,
        combine_entry,
        phase="combine",
    )
    print(f"Tuning config saved to: {dispatch_path} + {combine_path}")


class EpDispatchCombineTestCase:
    def __init__(
        self,
        rank,
        gpu_per_node,
        world_size,
        max_tokens,
        kernel_type,
        num_qp,
        quant_type="none",
        dtype=torch.bfloat16,
        hidden_dim=7168,
        combine_dtype=None,
        max_total_recv_tokens=0,
    ):
        self.rank = rank
        self.gpu_per_node = gpu_per_node
        self.world_size = world_size

        self.dispatch_data_type = dtype
        self.combine_data_type = combine_dtype if combine_dtype is not None else dtype
        self.dispatch_hidden_dim = (
            hidden_dim // 2 if _is_fp4x2_dtype(dtype) else hidden_dim
        )
        if _is_fp4x2_dtype(dtype) and not _is_fp4x2_dtype(self.combine_data_type):
            self.combine_hidden_dim = self.dispatch_hidden_dim * 2
        elif not _is_fp4x2_dtype(dtype) and _is_fp4x2_dtype(self.combine_data_type):
            self.combine_hidden_dim = self.dispatch_hidden_dim // 2
        else:
            self.combine_hidden_dim = self.dispatch_hidden_dim

        self.config = mori.ops.EpDispatchCombineConfig(
            data_type=dtype,
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=max(self.dispatch_hidden_dim, self.combine_hidden_dim),
            scale_dim=32,
            scale_type_size=4,
            max_num_inp_token_per_rank=max_tokens,
            num_experts_per_rank=256 // self.world_size,
            num_experts_per_token=8,
            warp_num_per_block=8,
            block_num=96,
            max_token_type_size=2,
            kernel_type=kernel_type_map[kernel_type],
            gpu_per_node=self.gpu_per_node,
            rdma_block_num=64,
            num_qp_per_pe=num_qp,
            quant_type=quant_type,
            max_total_recv_tokens=max_total_recv_tokens,
        )

    def setup(self):
        local_rank = self.rank % self.gpu_per_node
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)

        dist.init_process_group(
            backend="cpu:gloo",
            rank=self.rank,
            world_size=self.world_size,
        )

        print("init process group done")
        world_group = torch.distributed.group.WORLD
        assert world_group is not None

        print("process group ok")
        torch._C._distributed_c10d._register_process_group("default", world_group)
        mori.shmem.shmem_torch_process_group_init("default")

        print(f"I'm pe {mori.shmem.shmem_mype()} in {mori.shmem.shmem_npes()} pes")

        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(999)

    def cleanup(self):
        mori.shmem.shmem_finalize()
        dist.destroy_process_group()

    def _allgather_with_token_num_padding(self, input, max_token_num):
        shape = list(input.shape)

        pad_shape = shape.copy()
        pad_shape[0] = max_token_num - shape[0]

        target_shape = shape.copy()
        target_shape[0] = max_token_num

        output = [
            torch.zeros(
                target_shape,
                dtype=input.dtype,
                device=input.device,
            )
            for _ in range(self.world_size)
        ]
        padded_input = torch.cat(
            [
                input,
                torch.zeros(
                    pad_shape,
                    dtype=input.dtype,
                    device=input.device,
                ),
            ],
            0,
        )
        dist.all_gather(output, padded_input)
        return output

    def gen_test_data(self, max_num_token, use_max_token_num=False, only_my_rank=False):
        hidden_dim = self.dispatch_hidden_dim
        keep_ranks = {self.rank} if only_my_rank else set(range(self.world_size))

        # gen num_tokens
        if use_max_token_num:
            num_token = torch.tensor(
                [max_num_token for i in range(self.world_size)]
            ).to(self.device)
        else:
            num_token = torch.randint(
                1,
                max_num_token + 1,
                [self.world_size],
                generator=self.rng,
                device=self.device,
            )

        # gen indices - vectorized version for speed
        num_total_experts = self.config.num_experts_per_rank * self.config.world_size
        all_rank_indices = [None] * self.world_size
        for r in range(self.world_size):
            num_tok = num_token[r].item()
            random_vals = torch.rand(
                num_tok,
                num_total_experts,
                generator=self.rng,
                device=self.device,
            )
            indices = torch.argsort(random_vals, dim=1)[
                :, : self.config.num_experts_per_token
            ]
            if r in keep_ranks:
                all_rank_indices[r] = indices.to(torch.int32)
            del random_vals, indices

        # gen weights
        all_rank_weights = [None] * self.world_size
        for r in range(self.world_size):
            w = torch.rand(
                num_token[r],
                self.config.num_experts_per_token,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            if r in keep_ranks:
                all_rank_weights[r] = w
            else:
                del w

        # gen scales
        all_rank_scales = [None] * self.world_size
        for r in range(self.world_size):
            s = torch.rand(
                num_token[r],
                self.config.scale_dim,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            if r in keep_ranks:
                all_rank_scales[r] = s
            else:
                del s

        all_rank_input = [None] * self.world_size
        for r in range(self.world_size):
            data_fp32 = torch.randn(
                num_token[r],
                hidden_dim,
                dtype=torch.float32,
                generator=self.rng,
                device=self.device,
            )
            if _is_fp4x2_dtype(self.dispatch_data_type):
                data = torch.randint(
                    0,
                    256,
                    (num_token[r], hidden_dim),
                    dtype=torch.uint8,
                    generator=self.rng,
                    device=self.device,
                )
                data = data.view(_FP4_DTYPE)
            else:
                data = data_fp32.to(self.dispatch_data_type)
            del data_fp32
            if r in keep_ranks:
                all_rank_input[r] = data
            else:
                del data

        return (
            num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        )

    def _convert_for_combine(self, dispatch_output):
        """Convert dispatch output to combine dtype if cross-type."""
        if self.combine_data_type == self.dispatch_data_type:
            return dispatch_output
        if _is_fp4x2_dtype(self.dispatch_data_type):
            return unpack_fp4x2(dispatch_output, dtype=self.combine_data_type)
        return dispatch_output.to(self.combine_data_type)

    def count_token_num(self, all_rank_indices):
        # Per-rank counts
        rank_counts = torch.zeros(
            self.config.world_size, dtype=torch.int32, device=self.device
        )
        rank_counts_remote_recv = torch.zeros(
            self.config.world_size, dtype=torch.int32, device=self.device
        )
        rank_counts_remote_send = torch.zeros(
            self.config.world_size, dtype=torch.int32, device=self.device
        )

        for src_rank, indices in enumerate(all_rank_indices):
            src_node = src_rank // self.config.gpu_per_node

            # Map expert IDs to rank IDs
            token_ranks = (
                indices // self.config.num_experts_per_rank
            )  # [num_tokens, num_experts_per_token]

            # Deduplicate rank IDs per token
            unique_ranks_per_token = [torch.unique(row) for row in token_ranks]

            # For each token, update counts
            for ur in unique_ranks_per_token:
                rank_counts[ur] += 1  # All ranks that receive this token

                dst_nodes = {
                    dst_rank // self.config.gpu_per_node for dst_rank in ur.tolist()
                }

                for dst_rank in ur.tolist():
                    dst_node = dst_rank // self.config.gpu_per_node
                    if dst_node != src_node:
                        # Receiving side
                        rank_counts_remote_recv[dst_rank] += 1

                # Sending side (dedup by node: count once if token goes to a remote node)
                for dst_node in dst_nodes:
                    if dst_node != src_node:
                        rank_counts_remote_send[src_rank] += 1

        if self.config.rank == 0:
            print("Rank counts (deduplicated):", rank_counts)
        return rank_counts, rank_counts_remote_recv, rank_counts_remote_send

    def run_dispatch(
        self,
        op,
        token,
        weights,
        scales,
        indices,
        block_num=-1,
        rdma_block_num=-1,
        warp_per_block=-1,
    ):
        if op.config.kernel_type in (mori.ops.EpDispatchCombineKernelType.AsyncLL,):
            ret = op.dispatch(
                token,
                weights,
                scales,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
            op.dispatch_recv(block_num=block_num, warp_per_block=warp_per_block)
        else:
            ret = op.dispatch(
                token,
                weights,
                scales,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
        return ret

    def run_combine(
        self,
        op,
        token,
        weights,
        indices,
        block_num=-1,
        rdma_block_num=-1,
        warp_per_block=-1,
    ):
        if op.config.kernel_type in (mori.ops.EpDispatchCombineKernelType.AsyncLL,):
            ret = op.combine(
                token,
                weights,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
            op.combine_recv(block_num=block_num, warp_per_block=warp_per_block)
        else:
            ret = op.combine(
                token,
                weights,
                indices,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
        return ret

    def run_test_once(self, op, test_data, error_round, round):
        (
            all_rank_num_token,
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
        ) = self.run_dispatch(
            op,
            all_rank_input[self.rank],
            all_rank_weights[self.rank],
            all_rank_scales[self.rank],
            all_rank_indices[self.rank],
        )
        torch.cuda.synchronize()

        rank_counts, _, _ = self.count_token_num(all_rank_indices)

        src_token_pos = op.get_dispatch_src_token_pos().tolist()
        # max_num_token_to_send_per_rank = self.config.max_num_inp_token_per_rank
        recv_token_num = len(src_token_pos)

        # Check recv token num
        print(f"rank {self.rank} recv {recv_token_num} tokens")
        token_num_pass = rank_counts[self.rank] == recv_token_num
        if not token_num_pass:
            print(
                f"rank {self.rank} expected token num {rank_counts[self.rank]} got {recv_token_num}"
            )
            assert False

        # Check token equality
        for i, src_token_id in enumerate(src_token_pos):
            src_pe, src_tok_id = op.decode_send_flat_idx(src_token_id)
            if _is_fp4x2_dtype(self.config.data_type):
                is_pass = torch.equal(
                    dispatch_output[i].view(torch.uint8),
                    all_rank_input[src_pe][src_tok_id].view(torch.uint8),
                )
            else:
                is_pass = torch.equal(
                    dispatch_output[i], all_rank_input[src_pe][src_tok_id]
                )
            if not is_pass:
                print(
                    f"rank {self.rank} token {i} assert {is_pass} expected {all_rank_input[src_pe][src_tok_id].view(torch.uint8)} got {dispatch_output[i].view(torch.uint8)}"
                )
                assert False
            if dispatch_weights is not None:
                assert torch.equal(
                    dispatch_weights[i], all_rank_weights[src_pe][src_tok_id]
                )
            assert torch.equal(
                dispatch_indices[i], all_rank_indices[src_pe][src_tok_id]
            )
            assert torch.equal(dispatch_scales[i], all_rank_scales[src_pe][src_tok_id])

        if self.rank % self.gpu_per_node == 0:
            print(f"Node {self.rank // self.gpu_per_node} Dispatch Pass")

        # NOTE: weight combine not implemented yet
        if op.config.kernel_type in (mori.ops.EpDispatchCombineKernelType.AsyncLL,):
            dispatch_weights = None
        combine_input = self._convert_for_combine(dispatch_output)
        combine_output, combine_output_weight = self.run_combine(
            op, combine_input, dispatch_weights, all_rank_indices[self.rank]
        )
        torch.cuda.synchronize()
        combine_data_type = self.combine_data_type
        for i in range(all_rank_num_token[self.rank]):
            if _is_fp4x2_dtype(combine_data_type):
                continue
            pes = [
                (idx // self.config.num_experts_per_rank)
                for idx in all_rank_indices[self.rank][i].cpu().tolist()
            ]
            unique_pes = len(set(pes))
            unique_innode_pes = len(
                [
                    pe
                    for pe in set(pes)
                    if (pe // self.gpu_per_node == self.rank // self.gpu_per_node)
                ]
            )
            final_unique_pes = unique_pes
            if final_unique_pes == 0:
                continue

            inp = all_rank_input[self.rank][i]
            if _is_fp4x2_dtype(self.dispatch_data_type):
                inp_converted = unpack_fp4x2(
                    inp.unsqueeze(0), dtype=combine_data_type
                ).squeeze(0)
            else:
                inp_converted = inp.to(combine_data_type)

            got, expected = (
                combine_output[i],
                (inp_converted.to(torch.float32) * final_unique_pes).to(
                    combine_data_type
                ),
            )

            atol, rtol = 1e-2, 1e-2
            if getattr(self.config, "quant_type", "none") == "fp8_direct_cast":
                atol, rtol = 1e-1, 1e-1
            ok = torch.allclose(got.float(), expected.float(), atol=atol, rtol=rtol)
            if not ok:
                print(
                    self.rank,
                    f"token {i} pes {pes} unique pes {unique_pes} unique innode pes {unique_innode_pes}",
                )
                print(
                    f"{self.rank} got: ",
                    got,
                    f"{self.rank} expected: ",
                    expected,
                    all_rank_input[self.rank][i],
                )
                assert False
                # pass

            if dispatch_weights is not None:
                got_weight, expected_weight = (
                    combine_output_weight[i],
                    all_rank_weights[self.rank][i] * final_unique_pes,
                )
                weight_match = torch.allclose(
                    got_weight, expected_weight, atol=1e-5, rtol=1e-5
                )
                if not weight_match and self.config.rank == 0:
                    print(f"Weight mismatch for token {i}:")
                    print(
                        f"  indices[{i}]: {all_rank_indices[self.rank][i].cpu().tolist()}"
                    )
                    print(f"  pes: {pes}")
                    print(f"  unique_pes: {unique_pes}")
                    print(f"  got_weight: {got_weight}")
                    print(
                        f"  expected_weight (weights[{i}] * {unique_pes}): {expected_weight}"
                    )
                    print(f"  original weights[{i}]: {all_rank_weights[self.rank][i]}")
                    print(f"  diff: {torch.abs(got_weight - expected_weight)}")
                    print(
                        f"  max_diff: {torch.abs(got_weight - expected_weight).max()}"
                    )
                assert weight_match, f"Weight assertion failed for token {i}"
        if self.rank % self.gpu_per_node == 0:
            print(f"Node {self.rank // self.gpu_per_node} Combine Pass")

    def test_dispatch_combine(self):
        error_round = set()
        op = mori.ops.EpDispatchCombineOp(self.config)
        for i in range(500):
            if self.rank == 0:
                print(f"Round {i} begin")
            test_data = self.gen_test_data(
                max_num_token=self.config.max_num_inp_token_per_rank,
                use_max_token_num=False,
            )
            if self.rank == 0:
                print(f"Round {i} gen test_data done")
            self.run_test_once(op, test_data, error_round, i)
        print(
            "rank: ",
            self.rank,
            "error times: ",
            len(error_round),
            "appear round: ",
            error_round,
        )

        del op

    def stress_dispatch_combine(self):
        op = mori.ops.EpDispatchCombineOp(self.config)
        num_test_data = 128
        sync_interval = 128
        is_cross_type = self.combine_data_type != self.dispatch_data_type

        if self.rank == 0:
            print("Stress Test")
        test_data_list = [
            self.gen_test_data(
                max_num_token=self.config.max_num_inp_token_per_rank,
                use_max_token_num=False,
            )
            for i in range(num_test_data)
        ]
        for i in tqdm(range(5000)):
            (
                all_rank_num_token,
                all_rank_indices,
                all_rank_input,
                all_rank_weights,
                all_rank_scales,
            ) = test_data_list[i % num_test_data]
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = self.run_dispatch(
                op,
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
            )
            combine_input = self._convert_for_combine(dispatch_output)
            combine_output, combine_output_weight = self.run_combine(
                op, combine_input, None, all_rank_indices[self.rank]
            )
            if i % sync_interval == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()

        if is_cross_type:
            if self.rank == 0:
                print(
                    "Skipping CUDA Graph stress test (cross-type not supported in graph)"
                )
        else:
            if self.rank == 0:
                print("Stress Test with CUDA Graph")
            test_data = self.gen_test_data(
                max_num_token=self.config.max_num_inp_token_per_rank,
                use_max_token_num=False,
            )
            (
                all_rank_num_token,
                all_rank_indices,
                all_rank_input,
                all_rank_weights,
                all_rank_scales,
            ) = test_data
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                (
                    dispatch_output,
                    dispatch_weights,
                    dispatch_scales,
                    dispatch_indices,
                    dispatch_recv_num_token,
                ) = self.run_dispatch(
                    op,
                    all_rank_input[self.rank],
                    all_rank_weights[self.rank],
                    all_rank_scales[self.rank],
                    all_rank_indices[self.rank],
                )
                combine_output, combine_output_weight = self.run_combine(
                    op, dispatch_output, None, all_rank_indices[self.rank]
                )
            torch.cuda.synchronize()

            for i in tqdm(range(5000)):
                g.replay()
                torch.cuda.synchronize()
                time.sleep(0.0001)

        del op

    def run_bench_once(
        self,
        max_num_token,
        op,
        test_data,
        repeat=10,
        block_num=-1,
        rdma_block_num=-1,
        warp_per_block=-1,
    ):
        num_events = 3 * repeat + 1
        events = [torch.cuda.Event(enable_timing=True) for i in range(num_events)]

        (
            all_rank_num_token,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        warmup_rounds = 3
        for i in range(warmup_rounds):
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = self.run_dispatch(
                op,
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
            if i == warmup_rounds - 1:
                torch.cuda.synchronize()
                total_recv_num_token = dispatch_recv_num_token[0].item()
            combine_input = self._convert_for_combine(dispatch_output)
            combine_output, combine_output_weight = self.run_combine(
                op,
                combine_input,
                None,
                all_rank_indices[self.rank],
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
            torch.cuda.synchronize()
        total_rdma_recv_num_token = (
            self.config.max_num_inp_token_per_rank * self.config.world_size // 8
        )
        print(
            f"rank {self.rank} recv {total_recv_num_token} tokens {total_rdma_recv_num_token} rdma tokens"
        )

        if hasattr(mori.cpp, "get_debug_time_buf"):
            my_times = op.get_debug_time_buf()
            my_times.zero_()
            if hasattr(mori.cpp, "get_debug_time_offset"):
                my_offsets = op.get_debug_time_offset()
                my_offsets.zero_()

        torch.cuda.synchronize()
        dist.barrier()
        events[0].record()
        for i in range(repeat):
            (
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = self.run_dispatch(
                op,
                all_rank_input[self.rank],
                all_rank_weights[self.rank],
                all_rank_scales[self.rank],
                all_rank_indices[self.rank],
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
            events[3 * i + 1].record()
            combine_input = self._convert_for_combine(dispatch_output)
            events[3 * i + 2].record()
            combine_output, combine_output_weight = self.run_combine(
                op,
                combine_input,
                None,
                all_rank_indices[self.rank],
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
            )
            events[3 * i + 3].record()
        torch.cuda.synchronize()

        disp_element_size = all_rank_input[self.rank].element_size()
        disp_total_bytes = (
            total_recv_num_token * self.dispatch_hidden_dim * disp_element_size
        )
        comb_element_size = torch.tensor(
            [], dtype=self.combine_data_type
        ).element_size()
        comb_total_bytes = (
            total_recv_num_token * self.combine_hidden_dim * comb_element_size
        )
        ll_mode_scale = (
            max_num_token
            * self.config.num_experts_per_token
            / (total_recv_num_token + 1)
        )
        disp_total_rdma_bytes = (
            total_rdma_recv_num_token * self.dispatch_hidden_dim * disp_element_size
        )
        comb_total_rdma_bytes = (
            total_rdma_recv_num_token * self.combine_hidden_dim * comb_element_size
        )

        disp_duration_list = []
        comb_duration_list = []
        for i in range(repeat):
            disp_duration_list.append(events[3 * i].elapsed_time(events[3 * i + 1]))
            comb_duration_list.append(events[3 * i + 2].elapsed_time(events[3 * i + 3]))

        disp_rdma_bandwidth_list = [
            disp_total_rdma_bytes / (1000**3) / (t / (10**3))
            for t in disp_duration_list
        ]
        disp_bandwidth_list = [
            disp_total_bytes / (1000**3) / (t / (10**3)) for t in disp_duration_list
        ]

        comb_rdma_bandwidth_list = [
            comb_total_rdma_bytes / (1000**3) / (t / (10**3))
            for t in comb_duration_list
        ]
        comb_bandwidth_list = [
            comb_total_bytes / (1000**3) / (t / (10**3)) for t in comb_duration_list
        ]

        if hasattr(mori.cpp, "get_debug_time_buf"):
            output_filename = (
                f"trace_rank_{self.rank}_{time.strftime('%m%d_%H%M%S')}.json"
            )
            mori.kernel_profiler.export_to_perfetto(my_times, output_filename)
            if self.rank == 0:
                print(f"Profiling data exported to {output_filename}")

        return (
            disp_duration_list,
            disp_rdma_bandwidth_list,
            disp_bandwidth_list,
            comb_duration_list,
            comb_rdma_bandwidth_list,
            comb_bandwidth_list,
            ll_mode_scale,
        )

    def bench_dispatch_combine(
        self,
        max_num_token,
        block_num=-1,
        rdma_block_num=-1,
        warp_per_block=-1,
        skip_verify=False,
    ):
        op = mori.ops.EpDispatchCombineOp(self.config)
        test_data = self.gen_test_data(
            max_num_token=max_num_token,
            use_max_token_num=True,
            only_my_rank=skip_verify,
        )

        repeat = 10

        if not skip_verify:
            error_round = set()
            for i in range(1):
                if self.rank == 0:
                    print(f"WarmUp Round {i} begin")
                self.run_test_once(op, test_data, error_round, i)
            assert (
                len(error_round) == 0
            ), f"Warmup failed with errors in rounds: {error_round}"

        bench_result = self.run_bench_once(
            max_num_token,
            op,
            test_data,
            repeat,
            block_num=block_num,
            rdma_block_num=rdma_block_num,
            warp_per_block=warp_per_block,
        )
        ll_mode_scale = bench_result[-1]
        all_data, _ = self._all_gather_bench_data(bench_result)
        # all_data: (repeat, world_size, 6)
        # cols: d_rdma(0) d_xgmi(1) d_lat(2) c_rdma(3) c_xgmi(4) c_lat(5)

        if self.rank == 0:
            _labels = [
                (
                    "dispatch",
                    (
                        ("duration", 2, "µs"),
                        ("rdma bandwidth", 0, "GB/s"),
                        ("bandwidth", 1, "GB/s"),
                    ),
                ),
                (
                    "combine",
                    (
                        ("duration", 5, "µs"),
                        ("rdma bandwidth", 3, "GB/s"),
                        ("bandwidth", 4, "GB/s"),
                    ),
                ),
            ]
            for phase, cols in _labels:
                for i in range(all_data.shape[0]):
                    rd = all_data[i]
                    if cols is _labels[0][1]:
                        print(f"Round {i}")
                    for name, col, unit in cols:
                        print(
                            f"  {phase} {name} {rd[:, col].int().tolist()}"
                            f" avg {rd[:, col].mean():.2f} {unit}"
                        )

        if repeat == 1:
            return

        kept = all_data[1:]  # skip round 0
        disp_stats = self._build_phase_stats(kept, 0, 1, 2, ll_mode_scale)
        comb_stats = self._build_phase_stats(kept, 3, 4, 5, ll_mode_scale)

        disp_dtype_str = str(self.config.data_type).split(".")[-1]
        comb_dtype_str = str(self.combine_data_type).split(".")[-1]
        disp_title = f"Dispatch Performance ({disp_dtype_str})"
        comb_title = f"Combine Performance ({comb_dtype_str})"
        if self.combine_data_type != self.config.data_type:
            disp_elem = torch.tensor([], dtype=self.config.data_type).element_size()
            comb_elem = torch.tensor([], dtype=self.combine_data_type).element_size()
            disp_title += f" ~{max_num_token * self.dispatch_hidden_dim * disp_elem / (1024**2):.1f} MB/rank"
            comb_title += f" ~{max_num_token * self.combine_hidden_dim * comb_elem / (1024**2):.1f} MB/rank"

        if self.rank == 0:
            self._print_phase_table(disp_title, disp_stats)
            self._print_phase_table(comb_title, comb_stats)

        del op

        def _to_int_tuple(s):
            return tuple(round(v, 2) for v in s)

        return (
            _to_int_tuple(disp_stats["xgmi"]),
            _to_int_tuple(disp_stats["rdma"]),
            _to_int_tuple(disp_stats["ll"]),
            _to_int_tuple(disp_stats["lat"]),
        ), (
            _to_int_tuple(comb_stats["xgmi"]),
            _to_int_tuple(comb_stats["rdma"]),
            _to_int_tuple(comb_stats["ll"]),
            _to_int_tuple(comb_stats["lat"]),
        )

    # ------------------------------------------------------------------
    # Shared bench data collection — used by both bench and tuning
    # ------------------------------------------------------------------

    def _build_phase_stats(self, kept, col_rdma, col_xgmi, col_lat, ll_scale):
        """Compute (worst, best, avg) stats for one phase from gathered data.

        Returns dict with keys rdma, xgmi, ll, lat — each a (worst, best, avg) tuple.
        Uses the same ``_compute_stats`` as PrettyTable, so values are identical.
        """
        _s = self._compute_stats
        xgmi_s = _s(kept[:, :, col_xgmi])
        return {
            "rdma": _s(kept[:, :, col_rdma]),
            "xgmi": xgmi_s,
            "ll": tuple(v * ll_scale for v in xgmi_s),
            "lat": _s(kept[:, :, col_lat]),
        }

    @staticmethod
    def _print_phase_table(title, stats):
        """Print a PrettyTable for one phase from stats dict."""
        from prettytable import PrettyTable

        def _i(s):
            return tuple(round(v, 2) for v in s)

        tbl = PrettyTable()
        tbl.title = title
        tbl.field_names = [
            "Metrics",
            "RDMA Bandwidth (GB/s)",
            "XGMI Bandwidth (GB/s)",
            "LL Bandwidth (GB/s)",
            "Latency (us)",
        ]
        rdma, xgmi, ll, lat = (
            _i(stats["rdma"]),
            _i(stats["xgmi"]),
            _i(stats["ll"]),
            _i(stats["lat"]),
        )
        tbl.add_rows(
            [
                ["Best", rdma[1], xgmi[1], ll[1], lat[0]],
                ["Worst", rdma[0], xgmi[0], ll[0], lat[1]],
                ["Average", rdma[2], xgmi[2], ll[2], lat[2]],
            ]
        )
        print(tbl)

    _GATHER_COLS = (
        "disp_rdma",
        "disp_xgmi",
        "disp_lat",
        "comb_rdma",
        "comb_xgmi",
        "comb_lat",
    )

    def _all_gather_bench_data(self, bench_result):
        """Per-iteration all_gather from all ranks (packed into one call per round).

        Returns (data, ll_mode_scale) where *data* has shape
        ``(repeat, world_size, 6)`` with full float64 precision and columns
        [disp_rdma_bw, disp_xgmi_bw, disp_lat_us,
         comb_rdma_bw, comb_xgmi_bw, comb_lat_us].
        """
        (disp_dur, disp_rdma, disp_xgmi, comb_dur, comb_rdma, comb_xgmi, ll_scale) = (
            bench_result
        )
        repeat = len(disp_dur)

        rounds = []
        for i in range(repeat):
            local = torch.tensor(
                [
                    disp_rdma[i],
                    disp_xgmi[i],
                    disp_dur[i] * 1000,
                    comb_rdma[i],
                    comb_xgmi[i],
                    comb_dur[i] * 1000,
                ],
                dtype=torch.float64,
            )
            gathered = [
                torch.zeros(6, dtype=torch.float64) for _ in range(self.world_size)
            ]
            dist.all_gather(gathered, local)
            rounds.append(torch.stack(gathered))

        return torch.stack(rounds), ll_scale  # (repeat, world_size, 6)

    @staticmethod
    def _compute_stats(data_2d):
        """(worst, best, avg) from a (rounds, ranks) float tensor.

        avg = mean of per-round rank-means (grand mean, same as PrettyTable).
        """
        worst = data_2d.min().item()
        best = data_2d.max().item()
        avg = data_2d.mean(dim=1).mean().item()
        return (worst, best, avg)

    def tuning_dispatch_combine(
        self, max_num_token, save_tuning_config=None, skip_verify=False
    ):
        op = mori.ops.EpDispatchCombineOp(self.config)
        sm_count = torch.cuda.get_device_properties(
            self.rank % self.gpu_per_node
        ).multi_processor_count

        tuning_scope = os.environ.get("MORI_TUNING_SCOPE", "full")

        block_set = set()
        pow2 = 32
        while pow2 <= sm_count:
            block_set.add(pow2)
            pow2 <<= 1
        block_set.add(sm_count)
        block_list = sorted(block_set)

        if tuning_scope == "quick":
            warp_list = [4, 8, 16]
            _rdma_mode = "quick"
        else:
            warp_list = [4, 6, 8, 12, 16]
            _rdma_mode = "full"

        def rdma_candidates_for(bn):
            if _rdma_mode == "quick":
                return sorted(
                    set(v for v in [max(bn // 2, 1), bn * 2 // 3] if 1 <= v < bn)
                )
            return sorted(
                set(v for v in [bn // 4, bn // 2, bn * 2 // 3] if 1 <= v < bn)
            )

        is_ll_kernel = self.config.kernel_type in (
            mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
            mori.ops.EpDispatchCombineKernelType.AsyncLL,
        )
        bw_label = "LL BW" if is_ll_kernel else "RDMA BW"

        total_configs = sum(
            len(warp_list) * len(rdma_candidates_for(bn)) for bn in block_list
        )
        if self.rank == 0:
            print(
                f"SM count={sm_count}, tuning_scope={tuning_scope}, "
                f"skip_verify={skip_verify}\n"
                f"block_num candidates ({len(block_list)}): {block_list}\n"
                f"warp_per_block candidates: {warp_list}\n"
                f"BW metric: {bw_label}\n"
                f"Total configurations: {total_configs}"
            )

        best_disp_bw = 0
        best_comb_bw = 0
        best_disp_config = None
        best_comb_config = None
        _zero_stats = {
            "rdma": (0, 0, 0),
            "xgmi": (0, 0, 0),
            "ll": (0, 0, 0),
            "lat": (0, 0, 0),
        }
        best_disp_stats = dict(_zero_stats)
        best_comb_stats = dict(_zero_stats)

        test_data = self.gen_test_data(
            max_num_token=max_num_token,
            use_max_token_num=True,
            only_my_rank=skip_verify,
        )
        if not skip_verify:
            error_round = set()
            for wr_i in range(1):
                self.run_test_once(op, test_data, error_round, wr_i)
            assert len(error_round) == 0, f"Warmup failed: {error_round}"

        config_idx = 0
        for bn in block_list:
            for warp in warp_list:
                for rdma_bn in rdma_candidates_for(bn):
                    config_idx += 1
                    if self.rank == 0:
                        print(
                            f"\n{'=' * 60}\n"
                            f"[{config_idx}/{total_configs}] "
                            f"block_num={bn}, warp={warp}, rdma_block_num={rdma_bn}\n"
                            f"{'=' * 60}"
                        )
                    bench_result = self.run_bench_once(
                        max_num_token,
                        op,
                        test_data,
                        repeat=5,
                        block_num=bn,
                        rdma_block_num=rdma_bn,
                        warp_per_block=warp,
                    )
                    all_data, ll_scale = self._all_gather_bench_data(bench_result)
                    kept = all_data[1:]  # skip round 0, same as bench
                    # kept: (rounds, world_size, 6)
                    # cols: d_rdma, d_xgmi, d_lat, c_rdma, c_xgmi, c_lat

                    # Selection: per-rank avg across kept rounds, min rank
                    rank_means = kept.mean(dim=0)  # (world_size, 6)
                    if is_ll_kernel:
                        disp_bw = (rank_means[:, 1] * ll_scale).min().item()
                        comb_bw = (rank_means[:, 4] * ll_scale).min().item()
                    else:
                        disp_bw = rank_means[:, 0].min().item()
                        comb_bw = rank_means[:, 3].min().item()

                    # Stats via shared code (same as bench PrettyTable)
                    disp_stats = self._build_phase_stats(kept, 0, 1, 2, ll_scale)
                    comb_stats = self._build_phase_stats(kept, 3, 4, 5, ll_scale)

                    if disp_bw > best_disp_bw:
                        best_disp_bw = disp_bw
                        best_disp_config = (bn, warp, rdma_bn)
                        best_disp_stats = disp_stats
                    if comb_bw > best_comb_bw:
                        best_comb_bw = comb_bw
                        best_comb_config = (bn, warp, rdma_bn)
                        best_comb_stats = comb_stats

                    if self.rank == 0:
                        da, ca = disp_stats["xgmi"][2], comb_stats["xgmi"][2]
                        print(
                            f"  disp sel={disp_bw:.1f} xgmi={da:.1f}  "
                            f"comb sel={comb_bw:.1f} xgmi={ca:.1f}  "
                            f"(best disp={best_disp_bw:.1f} comb={best_comb_bw:.1f})"
                        )

        if self.rank == 0:
            disp_dtype_str = str(self.config.data_type).split(".")[-1]
            comb_dtype_str = str(self.combine_data_type).split(".")[-1]
            print(f"\n{'=' * 70}")
            print("Tuning Result (best config chosen by slowest-rank XGMI BW)")
            print(f"{'=' * 70}")
            for label, dtype_s, cfg, st in [
                ("Dispatch", disp_dtype_str, best_disp_config, best_disp_stats),
                ("Combine ", comb_dtype_str, best_comb_config, best_comb_stats),
            ]:
                self._print_phase_table(
                    f"{label} ({dtype_s}) block={cfg[0]} warp={cfg[1]} rdma={cfg[2]}",
                    st,
                )
            print(f"{'=' * 70}")

            if save_tuning_config and best_disp_config and best_comb_config:
                _save_internode_tuning_result(
                    save_tuning_config,
                    config=self.config,
                    max_num_token=max_num_token,
                    dispatch_hidden_dim=self.dispatch_hidden_dim,
                    combine_hidden_dim=self.combine_hidden_dim,
                    combine_data_type=self.combine_data_type,
                    best_disp_config=best_disp_config,
                    best_disp_bw=best_disp_bw,
                    best_disp_all_bw=best_disp_stats,
                    best_comb_config=best_comb_config,
                    best_comb_bw=best_comb_bw,
                    best_comb_all_bw=best_comb_stats,
                )

        del op

    def profile_dispatch_combine(self, max_num_token):
        op = mori.ops.EpDispatchCombineOp(self.config)
        test_data = self.gen_test_data(
            max_num_token=max_num_token, use_max_token_num=True
        )

        repeat = 3
        if not hasattr(mori.cpp, "get_debug_time_buf"):
            raise RuntimeError(
                "to use profiling command, re-compile MORI with ENABLE_PROFILER=ON"
            )

        self.run_bench_once(max_num_token, op, test_data, repeat)


def sweep_bench_dispatch_combine(
    local_rank,
    num_node,
    gpu_per_node,
    dtype,
    max_tokens,
    kernel_type,
    num_qp,
    sweep_token_interval,
    max_total_recv_tokens=0,
):
    world_size = num_node * gpu_per_node
    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank
    sweep_token_interval = int(sweep_token_interval)
    if sweep_token_interval <= 0:
        raise ValueError(f"sweep_token_interval must >= 1, got {sweep_token_interval}")
    test_case = EpDispatchCombineTestCase(
        global_rank,
        gpu_per_node,
        world_size,
        max_tokens,
        kernel_type,
        num_qp,
        dtype=dtype,
        max_total_recv_tokens=max_total_recv_tokens,
    )
    test_case.setup()

    num_iters = (max_tokens + sweep_token_interval - 1) // sweep_token_interval
    max_token_list = [i * sweep_token_interval for i in range(num_iters)]

    disp_lat_min_list = []
    disp_lat_max_list = []
    disp_lat_avg_list = []
    comb_lat_min_list = []
    comb_lat_max_list = []
    comb_lat_avg_list = []
    for max_token in max_token_list:
        if max_token == 0:
            max_token = 1
        disp_stats, comb_stats = test_case.bench_dispatch_combine(max_token)
        disp_bw, disp_rdma_bw, disp_ll_bw, disp_lat = disp_stats
        comb_bw, comb_rdma_bw, comb_ll_bw, comb_lat = comb_stats

        disp_lat_min_list.append(disp_lat[0])
        comb_lat_min_list.append(comb_lat[0])
        disp_lat_max_list.append(disp_lat[1])
        comb_lat_max_list.append(comb_lat[1])
        disp_lat_avg_list.append(disp_lat[2])
        comb_lat_avg_list.append(comb_lat[2])

    if local_rank == 0:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(max_token_list, disp_lat_min_list, label="Dispatch Min")
        plt.plot(max_token_list, comb_lat_min_list, label="Combine Min")
        plt.plot(max_token_list, disp_lat_max_list, label="Dispatch Max")
        plt.plot(max_token_list, comb_lat_max_list, label="Combine Max")
        plt.plot(max_token_list, disp_lat_avg_list, label="Dispatch Avg")
        plt.plot(max_token_list, comb_lat_avg_list, label="Combine Avg")

        plt.xticks([i * 16 for i in range(max_tokens // 16)])
        plt.title("Dispatch / Combine Latency (us)")
        plt.xlabel("# of Tokens")
        plt.ylabel("Latency (us)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("dispatch_combine_perf.png", dpi=300, bbox_inches="tight")
        test_case.cleanup()


def test_dispatch_combine(
    local_rank,
    num_node,
    gpu_per_node,
    dtype,
    max_tokens,
    kernel_type,
    num_qp,
    quant_type="none",
    cmd="test",
    sweep_token_interval=64,
    combine_dtype=None,
    block_num=-1,
    rdma_block_num=-1,
    warp_per_block=-1,
    max_total_recv_tokens=0,
    hidden_dim=7168,
    save_tuning_config=None,
    skip_verify=False,
):
    world_size = num_node * gpu_per_node
    node_rank = int(os.environ["RANK"])
    global_rank = node_rank * gpu_per_node + local_rank

    if cmd in ("test", "bench", "stress", "profile", "tuning"):
        test_case = EpDispatchCombineTestCase(
            global_rank,
            gpu_per_node,
            world_size,
            max_tokens,
            kernel_type,
            num_qp,
            quant_type,
            dtype,
            hidden_dim=hidden_dim,
            combine_dtype=combine_dtype,
            max_total_recv_tokens=max_total_recv_tokens,
        )
        test_case.setup()
        if cmd == "test":
            test_case.test_dispatch_combine()
        elif cmd == "bench":
            test_case.bench_dispatch_combine(
                max_tokens,
                block_num=block_num,
                rdma_block_num=rdma_block_num,
                warp_per_block=warp_per_block,
                skip_verify=skip_verify,
            )
        elif cmd == "stress":
            test_case.stress_dispatch_combine()
        elif cmd == "profile":
            test_case.profile_dispatch_combine(max_tokens)
        elif cmd == "tuning":
            test_case.tuning_dispatch_combine(
                max_tokens,
                save_tuning_config=save_tuning_config,
                skip_verify=skip_verify,
            )
        test_case.cleanup()
    elif cmd == "sweep_bench":
        sweep_bench_dispatch_combine(
            local_rank,
            num_node,
            gpu_per_node,
            dtype,
            max_tokens,
            kernel_type,
            num_qp,
            sweep_token_interval,
            max_total_recv_tokens=max_total_recv_tokens,
        )
    else:
        raise ValueError(f"unsupported command: {cmd}")


_DATA_TYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8_e4m3_fnuz": torch.float8_e4m3fnuz,
    "fp8_e4m3": torch.float8_e4m3fn,
}
if hasattr(torch, "float4_e2m1fn_x2"):
    _DATA_TYPE_MAP["fp4"] = torch.float4_e2m1fn_x2


parser = argparse.ArgumentParser(description="dispatch/combine internode test")
parser.add_argument(
    "--cmd",
    type=str,
    default="test",
    choices=["test", "bench", "stress", "sweep_bench", "profile", "tuning"],
    help="Available subcommands: test, bench, stress, sweep_bench, profile, tuning",
)
parser.add_argument(
    "--dtype",
    type=str,
    default="bf16",
    choices=list(_DATA_TYPE_MAP.keys()),
    help="Data type of dispatch / combine",
)
parser.add_argument(
    "--combine-dtype",
    type=str,
    default=None,
    choices=["bf16", "fp8_e4m3_fnuz", "fp8_e4m3"],
    help="Data type for combine phase. Defaults to same as --dtype.",
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=4096,
    help="Maximum number of input tokens per rank (default: 4096)",
)
parser.add_argument(
    "--sweep-token-interval",
    type=int,
    default=2,
    help="Number of token interval when sweep bench",
)
parser.add_argument(
    "--kernel-type",
    type=str,
    default="v1",
    help="Type of kernel to test",
    choices=["v0", "v1", "v1_ll", "async_ll"],
)
parser.add_argument(
    "--num-qp",
    type=int,
    default=1,
    help="Number of qp per processing endpoint",
)
parser.add_argument(
    "--quant-type",
    type=str,
    default="none",
    choices=["none", "fp8_direct_cast"],
    help=(
        "Quantization method used inside Combine. "
        "'fp8_direct_cast' is the current BF16<->FP8 direct cast path."
    ),
)
parser.add_argument(
    "--block-num",
    type=int,
    default=None,
    help="Override block_num for bench mode.",
)
parser.add_argument(
    "--warp-per-block",
    type=int,
    default=None,
    help="Override warp_per_block for bench mode.",
)
parser.add_argument(
    "--rdma-block-num",
    type=int,
    default=None,
    help="Override rdma_block_num for bench mode.",
)
parser.add_argument(
    "--max-recv-total-tokens",
    type=int,
    default=0,
    help="Maximum total number of received tokens across all ranks (default: 0, meaning no limit)",
)
parser.add_argument(
    "--hidden-dim",
    type=int,
    default=7168,
    help="Base hidden dimension for the model (default: 7168)",
)
parser.add_argument(
    "--save-tuning-config",
    type=str,
    default=None,
    help=(
        "Path to save tuning results as JSON config. "
        "Use 'auto' to auto-generate filename based on GPU/dtype/EP."
    ),
)
parser.add_argument(
    "--skip-verify",
    action="store_true",
    default=False,
    help="Skip correctness verification in bench mode to reduce GPU memory usage.",
)
args_cli = parser.parse_args()

if __name__ == "__main__":
    gpu_per_node = os.environ.get("GPU_PER_NODE", None)
    gpu_per_node = int(gpu_per_node) if gpu_per_node is not None else 8
    num_node = int(os.environ["WORLD_SIZE"])

    combine_dtype_str = (
        args_cli.combine_dtype if args_cli.combine_dtype else args_cli.dtype
    )
    dispatch_dtype = _DATA_TYPE_MAP[args_cli.dtype]
    combine_dtype = _DATA_TYPE_MAP[combine_dtype_str]

    world_size = num_node * gpu_per_node
    torch.multiprocessing.spawn(
        test_dispatch_combine,
        args=(
            num_node,
            gpu_per_node,
            dispatch_dtype,
            args_cli.max_tokens,
            args_cli.kernel_type,
            args_cli.num_qp,
            args_cli.quant_type,
            args_cli.cmd,
            args_cli.sweep_token_interval,
            combine_dtype,
            args_cli.block_num if args_cli.block_num is not None else -1,
            args_cli.rdma_block_num if args_cli.rdma_block_num is not None else -1,
            args_cli.warp_per_block if args_cli.warp_per_block is not None else -1,
            args_cli.max_recv_total_tokens,
            args_cli.hidden_dim,
            args_cli.save_tuning_config,
            args_cli.skip_verify,
        ),
        nprocs=gpu_per_node,
        join=True,
    )
