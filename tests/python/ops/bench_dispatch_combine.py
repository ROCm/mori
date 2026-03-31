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
from collections import namedtuple
from tests.python.ops.dispatch_combine_test_utils import (
    _is_fp4x2_dtype,
    unpack_fp4x2,
    EpDispatchCombineTestCase,
)
from tests.python.utils import TorchDistContext, get_free_port
import torch
import torch.distributed as dist
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "6G")


class EpDispatchCombineBenchmark(EpDispatchCombineTestCase):
    def __init__(
        self,
        config,
        combine_data_type=None,
        combine_hidden_dim=None,
        dispatch_hidden_dim=None,
    ):
        super().__init__(config)
        self.combine_data_type = (
            combine_data_type if combine_data_type is not None else config.data_type
        )
        self.combine_hidden_dim = (
            combine_hidden_dim if combine_hidden_dim is not None else config.hidden_dim
        )
        self.dispatch_hidden_dim = (
            dispatch_hidden_dim
            if dispatch_hidden_dim is not None
            else config.hidden_dim
        )

    def gen_test_data(self):
        saved = self.config.hidden_dim
        self.config.hidden_dim = self.dispatch_hidden_dim
        result = super().gen_test_data(use_max_token_num=True)
        self.config.hidden_dim = saved
        return result

    def _get_combine_input(self, op, dispatch_output):
        """Return the tensor to pass as combine input.

        For external-input-buffer mode (non-zero-copy): returns the data tensor
        directly (converted if cross-type).
        For zero-copy mode: returns the registered shmem buffer; the caller
        is responsible for copying converted data into it.
        """
        if self.config.use_external_inp_buf:
            if self.combine_data_type != self.config.data_type:
                if _is_fp4x2_dtype(self.config.data_type):
                    return unpack_fp4x2(dispatch_output, dtype=self.combine_data_type)
                return dispatch_output.to(self.combine_data_type)
            return dispatch_output
        return op.get_registered_combine_input_buffer(
            self.combine_data_type, hidden_dim=self.combine_hidden_dim
        )

    def _capture_split_graphs(
        self,
        op,
        test_data,
        dispatch_block_num,
        dispatch_warp_per_block,
        combine_block_num,
        combine_warp_per_block,
    ):
        (
            _,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        dispatch_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(dispatch_graph):
            (
                dispatch_output,
                _,
                _,
                dispatch_indices,
                dispatch_recv_num_token,
            ) = op.dispatch(
                all_rank_input[self.config.rank],
                all_rank_weights[self.config.rank],
                # None,
                all_rank_scales[self.config.rank],
                all_rank_indices[self.config.rank],
                block_num=dispatch_block_num,
                warp_per_block=dispatch_warp_per_block,
            )

        combine_input = self._get_combine_input(op, dispatch_output)

        combine_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(combine_graph):
            combine_output, _ = op.combine(
                combine_input,
                # dispatch_weights,
                None,
                dispatch_indices,
                block_num=combine_block_num,
                warp_per_block=combine_warp_per_block,
            )
        self.sync()

        return dispatch_graph, combine_graph

    def _capture_e2e_graph(
        self,
        op,
        test_data,
        dispatch_block_num,
        dispatch_warp_per_block,
        combine_block_num,
        combine_warp_per_block,
        total_recv_num_token,
    ):
        (
            _,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        e2e_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(e2e_graph):
            (
                e2e_dispatch_output,
                _,
                _,
                e2e_dispatch_indices,
                e2e_dispatch_recv_num_token,
            ) = op.dispatch(
                all_rank_input[self.config.rank],
                all_rank_weights[self.config.rank],
                # None,
                all_rank_scales[self.config.rank],
                all_rank_indices[self.config.rank],
                block_num=dispatch_block_num,
                warp_per_block=dispatch_warp_per_block,
            )
            e2e_combine_arg = self._get_combine_input(op, e2e_dispatch_output)
            e2e_combine_output, _ = op.combine(
                e2e_combine_arg,
                # dispatch_weights,
                None,
                e2e_dispatch_indices,
                block_num=combine_block_num,
                warp_per_block=combine_warp_per_block,
            )
        self.sync()

        return e2e_graph

    def run_once_test(
        self,
        op,
        test_data,
        dispatch_block_num,
        dispatch_warp_per_block,
        combine_block_num,
        combine_warp_per_block,
    ):
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
        ) = op.dispatch(
            all_rank_input[self.config.rank],
            all_rank_weights[self.config.rank],
            # None,
            all_rank_scales[self.config.rank],
            all_rank_indices[self.config.rank],
            block_num=dispatch_block_num,
            warp_per_block=dispatch_warp_per_block,
        )
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

        combine_input = self._get_combine_input(op, dispatch_output)
        if not self.config.use_external_inp_buf:
            if self.combine_data_type != self.config.data_type:
                if _is_fp4x2_dtype(self.config.data_type):
                    converted = unpack_fp4x2(
                        dispatch_output[:total_recv_num_token],
                        dtype=self.combine_data_type,
                    )
                else:
                    converted = dispatch_output[:total_recv_num_token].to(
                        self.combine_data_type
                    )
                combine_input[:total_recv_num_token, :].copy_(converted)
            else:
                combine_input[:total_recv_num_token, :].copy_(
                    dispatch_output[:total_recv_num_token, :]
                )

        combine_output, _ = op.combine(
            combine_input,
            None,
            dispatch_indices,
            block_num=combine_block_num,
            warp_per_block=combine_warp_per_block,
        )
        self.check_combine_result(
            op,
            test_data,
            combine_output,
            combine_data_type=self.combine_data_type,
        )
        self.sync()
        return total_recv_num_token

    def run_once_bench(
        self,
        op,
        test_data,
        dispatch_block_num,
        dispatch_warp_per_block,
        combine_block_num,
        combine_warp_per_block,
        graph_replay_iters=10,
    ):
        (
            _,
            all_rank_indices,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        if graph_replay_iters <= 0:
            raise ValueError("graph_replay_iters must be greater than 0")

        # Run one real pair first so the dynamic recv-token count is available
        # before we build the benchmark graphs.
        total_recv_num_token = self.run_once_test(
            op,
            test_data,
            dispatch_block_num,
            dispatch_warp_per_block,
            combine_block_num,
            combine_warp_per_block,
        )

        # Split dispatch/combine into two graphs so we can time each replay
        # separately with host-side events.
        dispatch_graph, combine_graph = self._capture_split_graphs(
            op,
            test_data,
            dispatch_block_num,
            dispatch_warp_per_block,
            combine_block_num,
            combine_warp_per_block,
        )

        is_cross_type = self.combine_data_type != self.config.data_type

        # e2e graph not supported for cross-type (graph capture cannot
        # allocate memory for type conversion).
        e2e_graph = None
        if not is_cross_type:
            e2e_graph = self._capture_e2e_graph(
                op,
                test_data,
                dispatch_block_num,
                dispatch_warp_per_block,
                combine_block_num,
                combine_warp_per_block,
                total_recv_num_token,
            )

        round_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)
        ]
        mid_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)
        ]
        round_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)
        ]
        e2e_start_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)
        ]
        e2e_end_events = [
            torch.cuda.Event(enable_timing=True) for _ in range(graph_replay_iters)
        ]

        for i in range(graph_replay_iters):
            round_start_events[i].record()
            dispatch_graph.replay()
            mid_events[i].record()
            combine_graph.replay()
            round_end_events[i].record()

            if e2e_graph is not None:
                e2e_start_events[i].record()
                e2e_graph.replay()
                e2e_end_events[i].record()

        torch.cuda.synchronize()
        disp_duration = (
            sum(
                start_event.elapsed_time(end_event)
                for start_event, end_event in zip(round_start_events, mid_events)
            )
            / graph_replay_iters
        )
        comb_duration = (
            sum(
                start_event.elapsed_time(end_event)
                for start_event, end_event in zip(mid_events, round_end_events)
            )
            / graph_replay_iters
        )
        if e2e_graph is not None:
            e2e_duration = (
                sum(
                    start_event.elapsed_time(end_event)
                    for start_event, end_event in zip(e2e_start_events, e2e_end_events)
                )
                / graph_replay_iters
            )
        else:
            e2e_duration = -1.0
        self.sync()

        disp_element_size = all_rank_input[self.config.rank].element_size()
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
            self.config.max_num_inp_token_per_rank
            * self.config.num_experts_per_token
            / (total_recv_num_token + 0.01)
        )
        disp_bandwidth = disp_total_bytes / (1000**3) / (disp_duration / (10**3))
        comb_bandwidth = comb_total_bytes / (1000**3) / (comb_duration / (10**3))

        return (
            disp_duration,
            comb_duration,
            e2e_duration,
            disp_bandwidth,
            comb_bandwidth,
            disp_total_bytes,
            comb_total_bytes,
            ll_mode_scale,
        )

    def run(
        self,
        op,
        dispatch_block_num,
        dispatch_warp_per_block,
        combine_block_num,
        combine_warp_per_block,
        warmup=1,
        iters=10,
        graph_replay_iters=10,
    ):
        test_data = self.gen_test_data()
        for _ in range(warmup):
            self.run_once_test(
                op,
                test_data,
                dispatch_block_num,
                dispatch_warp_per_block,
                combine_block_num,
                combine_warp_per_block,
            )

        disp_duration_us_list = []
        disp_bandwidth_GB_list = []
        comb_duration_us_list = []
        comb_bandwidth_GB_list = []
        e2e_duration_us_list = []
        disp_avg_bytes_MB_list = []
        comb_avg_bytes_MB_list = []

        test_data_list = [self.gen_test_data() for i in range(iters)]

        for i in range(iters):
            self.sync()
            (
                disp_dur,
                comb_dur,
                e2e_dur,
                disp_bw,
                comb_bw,
                disp_total_bytes,
                comb_total_bytes,
                ll_mode_scale,
            ) = self.run_once_bench(
                op,
                test_data_list[i],
                dispatch_block_num,
                dispatch_warp_per_block,
                combine_block_num,
                combine_warp_per_block,
                graph_replay_iters=graph_replay_iters,
            )

            disp_dur_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            disp_bw_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            comb_dur_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            comb_bw_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            e2e_dur_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            disp_bytes_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            comb_bytes_list = [torch.zeros(1) for _ in range(self.config.world_size)]

            dist.all_gather(disp_dur_list, torch.tensor([disp_dur * 1000]))
            dist.all_gather(disp_bw_list, torch.tensor([disp_bw]))
            dist.all_gather(comb_dur_list, torch.tensor([comb_dur * 1000]))
            dist.all_gather(comb_bw_list, torch.tensor([comb_bw]))
            dist.all_gather(e2e_dur_list, torch.tensor([e2e_dur * 1000]))
            dist.all_gather(
                disp_bytes_list, torch.tensor([disp_total_bytes / (1024**2)])
            )
            dist.all_gather(
                comb_bytes_list, torch.tensor([comb_total_bytes / (1024**2)])
            )

            disp_duration_us_list.append([int(t.item()) for t in disp_dur_list])
            disp_bandwidth_GB_list.append([int(t.item()) for t in disp_bw_list])
            comb_duration_us_list.append([int(t.item()) for t in comb_dur_list])
            comb_bandwidth_GB_list.append([int(t.item()) for t in comb_bw_list])
            e2e_duration_us_list.append([int(t.item()) for t in e2e_dur_list])
            disp_avg_bytes_MB_list.append(
                int(torch.tensor(disp_bytes_list).mean().item())
            )
            comb_avg_bytes_MB_list.append(
                int(torch.tensor(comb_bytes_list).mean().item())
            )

        max_disp_algo_bw = 0
        max_comb_algo_bw = 0
        min_disp_latency_us = float("inf")
        min_comb_latency_us = float("inf")
        for i in range(iters):
            disp_algo_bw = sum(disp_bandwidth_GB_list[i]) / self.config.world_size
            comb_algo_bw = sum(comb_bandwidth_GB_list[i]) / self.config.world_size
            max_disp_algo_bw = max(max_disp_algo_bw, disp_algo_bw)
            max_comb_algo_bw = max(max_comb_algo_bw, comb_algo_bw)
            disp_max_lat = max(disp_duration_us_list[i])
            comb_max_lat = max(comb_duration_us_list[i])
            min_disp_latency_us = min(min_disp_latency_us, disp_max_lat)
            min_comb_latency_us = min(min_comb_latency_us, comb_max_lat)

        if self.config.rank == 0:
            print("Dispatch result:")
            for i, duration_us in enumerate(disp_duration_us_list):
                algo_bw = sum(disp_bandwidth_GB_list[i]) / self.config.world_size
                lat = max(duration_us)
                print(
                    f"Round {i} duration(us) {duration_us} "
                    f"bandwidth(GB/s) {disp_bandwidth_GB_list[i]} "
                    f"avg bytes(MB) {disp_avg_bytes_MB_list[i]} lat {lat} bw {algo_bw} / {algo_bw * ll_mode_scale:.2f}"
                )

            print()
            print("Combine result:")
            for i, duration_us in enumerate(comb_duration_us_list):
                algo_bw = sum(comb_bandwidth_GB_list[i]) / self.config.world_size
                lat = max(duration_us)
                print(
                    f"Round {i} duration(us) {duration_us} "
                    f"bandwidth(GB/s) {comb_bandwidth_GB_list[i]} "
                    f"avg bytes(MB) {comb_avg_bytes_MB_list[i]} lat {lat} bw {algo_bw} / {algo_bw * ll_mode_scale:.2f}"
                )

            print()
            if e2e_duration_us_list and e2e_duration_us_list[0][0] >= 0:
                print("End-to-end result:")
                print(
                    "Note: e2e is one full-graph replay; separate results use two graph replays, so sep total is usually higher."
                )
                for i, duration_us in enumerate(e2e_duration_us_list):
                    print(f"Round {i} e2e(us) {duration_us} ")
            else:
                print("End-to-end result: skipped (cross-type)")

        return (
            max_disp_algo_bw,
            max_comb_algo_bw,
            min_disp_latency_us,
            min_comb_latency_us,
        )

    def stress(
        self,
        op,
        dispatch_block_num,
        dispatch_warp_per_block,
        combine_block_num,
        combine_warp_per_block,
    ):
        if self.combine_data_type != self.config.data_type:
            raise ValueError(
                "Stress mode does not support cross-type dispatch/combine. "
                "Use --cmd bench or --cmd tuning instead."
            )
        test_data = self.gen_test_data()
        total_recv_num_token = self.run_once_test(
            op,
            test_data,
            dispatch_block_num,
            dispatch_warp_per_block,
            combine_block_num,
            combine_warp_per_block,
        )
        g = self._capture_e2e_graph(
            op,
            test_data,
            dispatch_block_num,
            dispatch_warp_per_block,
            combine_block_num,
            combine_warp_per_block,
            total_recv_num_token,
        )
        torch.cuda.synchronize()
        for i in range(135):
            if self.config.rank == 0:
                print(f"Round {i} begin")
            g.replay()
            torch.cuda.synchronize()


def _save_intranode_tuning_result(
    config_path,
    world_size,
    max_num_inp_token_per_rank,
    dispatch_hidden_dim,
    combine_hidden_dim,
    data_type,
    combine_data_type,
    quant_type,
    best_disp_config,
    best_disp_bw,
    best_comb_config,
    best_comb_bw,
    zero_copy=True,
):
    from pathlib import Path
    from mori.ops.tuning_config import (
        TuningConfigManager,
        dtype_to_config_str,
        build_config_filename,
        quant_type_to_config_str,
        detect_gpu_model,
    )
    from mori.jit.config import detect_gpu_arch

    gpu_arch = detect_gpu_arch()
    gpu_model = detect_gpu_model()
    disp_dtype_str = dtype_to_config_str(data_type)
    comb_dtype_str = dtype_to_config_str(combine_data_type)
    qt_str = quant_type_to_config_str(quant_type)

    metadata = {
        "gpu_arch": gpu_arch,
        "gpu_model": gpu_model,
        "kernel_type": "IntraNode",
        "ep_size": world_size,
        "quant_type": qt_str,
    }

    dispatch_entry = {
        "dtype": disp_dtype_str,
        "num_tokens": max_num_inp_token_per_rank,
        "hidden_dim": dispatch_hidden_dim,
        "block_num": best_disp_config[0],
        "rdma_block_num": 0,
        "warp_per_block": best_disp_config[1],
        "bandwidth_gbps": round(best_disp_bw, 2),
    }

    combine_entry = {
        "dtype": comb_dtype_str,
        "num_tokens": max_num_inp_token_per_rank,
        "hidden_dim": combine_hidden_dim,
        "zero_copy": zero_copy,
        "block_num": best_comb_config[0],
        "rdma_block_num": 0,
        "warp_per_block": best_comb_config[1],
        "bandwidth_gbps": round(best_comb_bw, 2),
    }

    if config_path == "auto":
        repo_tuning_dir = (
            Path(__file__).resolve().parents[3]
            / "python"
            / "mori"
            / "ops"
            / "tuning_configs"
        )
        filename = build_config_filename(
            gpu_arch,
            "IntraNode",
            world_size,
            qt_str,
            gpu_model,
        )
        config_path = str(repo_tuning_dir / filename)

    TuningConfigManager.save_tuning_result(
        config_path, metadata, dispatch_entry, combine_entry
    )
    print(f"Tuning config saved to: {config_path}")


LaunchConfig = namedtuple(
    "LaunchConfig",
    [
        "dispatch_block_num",
        "dispatch_warp_per_block",
        "combine_block_num",
        "combine_warp_per_block",
    ],
)


def _get_default_launch_config(
    world_size,
    max_num_inp_token_per_rank,
    use_external_inp_buf,
):
    zero_copy = not use_external_inp_buf
    if world_size <= 4:
        if max_num_inp_token_per_rank > 128:
            return (
                LaunchConfig(768, 8, 72, 4)
                if zero_copy
                else LaunchConfig(768, 8, 256, 14)
            )
        if max_num_inp_token_per_rank > 64:
            return (
                LaunchConfig(216, 6, 72, 4)
                if zero_copy
                else LaunchConfig(216, 6, 224, 8)
            )
        return (
            LaunchConfig(223, 6, 72, 4) if zero_copy else LaunchConfig(223, 6, 224, 4)
        )

    if max_num_inp_token_per_rank > 1024:
        return (
            LaunchConfig(80, 16, 80, 4) if zero_copy else LaunchConfig(80, 16, 80, 16)
        )
    return LaunchConfig(64, 16, 64, 4) if zero_copy else LaunchConfig(64, 16, 64, 16)


def _bench_dispatch_combine(
    rank,
    world_size,
    port,
    max_num_inp_token_per_rank,
    data_type,
    hidden_dim,
    scale_dim,
    scale_type_size,
    num_experts_per_rank,
    num_experts_per_token,
    cmd="bench",
    zero_copy=1,
    quant_type="none",
    dispatch_block_num_arg=None,
    dispatch_warp_per_block_arg=None,
    combine_block_num_arg=None,
    combine_warp_per_block_arg=None,
    combine_data_type=None,
    max_total_recv_tokens=0,
    save_tuning_config=None,
):
    if combine_data_type is None:
        combine_data_type = data_type

    if _is_fp4x2_dtype(data_type) and not _is_fp4x2_dtype(combine_data_type):
        combine_hidden_dim = hidden_dim * 2
    elif not _is_fp4x2_dtype(data_type) and _is_fp4x2_dtype(combine_data_type):
        combine_hidden_dim = hidden_dim // 2
    else:
        combine_hidden_dim = hidden_dim

    if quant_type == "fp8_direct_cast" and combine_data_type is not torch.bfloat16:
        raise ValueError(
            "fp8_direct_cast quant requires combine dtype to be bfloat16, "
            f"got {combine_data_type}"
        )

    config = mori.ops.EpDispatchCombineConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=max(hidden_dim, combine_hidden_dim),
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_token_type_size=2,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        max_total_recv_tokens=max_total_recv_tokens,
        warp_num_per_block=16,
        block_num=80,
        use_external_inp_buf=not zero_copy,  # zero-copy mode requires use_external_inp_buf=False
        gpu_per_node=world_size,
        quant_type=quant_type,
    )
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        mori.shmem.shmem_torch_process_group_init("default")
        op = mori.ops.EpDispatchCombineOp(config)
        benchmark = EpDispatchCombineBenchmark(
            config,
            combine_data_type=combine_data_type,
            combine_hidden_dim=combine_hidden_dim,
            dispatch_hidden_dim=hidden_dim,
        )

        (
            dispatch_block_num,
            dispatch_warp_per_block,
            combine_block_num,
            combine_warp_per_block,
        ) = _get_default_launch_config(
            world_size=world_size,
            max_num_inp_token_per_rank=max_num_inp_token_per_rank,
            use_external_inp_buf=config.use_external_inp_buf,
        )

        if cmd != "tuning":
            if dispatch_block_num_arg is not None:
                dispatch_block_num = dispatch_block_num_arg
            if dispatch_warp_per_block_arg is not None:
                dispatch_warp_per_block = dispatch_warp_per_block_arg
            if combine_block_num_arg is not None:
                combine_block_num = combine_block_num_arg
            if combine_warp_per_block_arg is not None:
                combine_warp_per_block = combine_warp_per_block_arg

        if cmd == "bench":
            if rank == 0:
                print(f"\n{'=' * 60}")
                print(
                    f"Benchmarking with dispatch_block_num={dispatch_block_num}, dispatch_warp_per_block={dispatch_warp_per_block} combine_block_num={combine_block_num}, combine_warp_per_block={combine_warp_per_block}"
                )
                print(f"{'=' * 60}")
            benchmark.run(
                op,
                dispatch_block_num=dispatch_block_num,
                dispatch_warp_per_block=dispatch_warp_per_block,
                combine_block_num=combine_block_num,
                combine_warp_per_block=combine_warp_per_block,
            )

        elif cmd == "stress":
            # Stress test
            if rank == 0:
                print(f"\n{'=' * 60}")
                print(
                    f"Stress testing with dispatch_block_num={dispatch_block_num}, dispatch_warp_per_block={dispatch_warp_per_block} combine_block_num={combine_block_num}, combine_warp_per_block={combine_warp_per_block}"
                )
                print(f"{'=' * 60}")
            benchmark.stress(
                op,
                dispatch_block_num=dispatch_block_num,
                dispatch_warp_per_block=dispatch_warp_per_block,
                combine_block_num=combine_block_num,
                combine_warp_per_block=combine_warp_per_block,
            )

        elif cmd == "tuning":
            if rank == 0 and any(
                x is not None
                for x in (
                    dispatch_block_num_arg,
                    dispatch_warp_per_block_arg,
                    combine_block_num_arg,
                    combine_warp_per_block_arg,
                )
            ):
                print(
                    "Warning: dispatch/combine block/warp arguments are ignored when --cmd tuning"
                )
            sm_count = torch.cuda.get_device_properties(rank).multi_processor_count

            # Build full dispatch candidate set (includes over-subscribe)
            max_disp_block_num = max(sm_count * 4, 320)
            all_block_set = set()
            all_block_set.update(range(32, sm_count + 1, 8))
            pow2 = 32
            while pow2 <= max_disp_block_num:
                all_block_set.add(pow2)
                pow2 <<= 1
            for anchor in [224, 256]:
                for delta in [-32, -16, -8, -4, -1, 0, 1, 4, 8, 16, 32]:
                    v = anchor + delta
                    if 32 <= v <= max_disp_block_num:
                        all_block_set.add(v)
            for mult in [1, 2, 3, 4]:
                all_block_set.add(sm_count * mult)

            common_block_list = sorted(v for v in all_block_set if v <= sm_count)
            # Over-subscribe (block_num > SM count) is disabled by default due to
            # hang risk with inter-block synchronization. Uncomment to enable:
            # extra_disp_block_list = sorted(v for v in all_block_set if v > sm_count)
            extra_disp_block_list = []

            warp_per_block_list = [4, 5, 6, 8, 10, 12, 14, 15, 16]

            if rank == 0:
                print(
                    f"SM count={sm_count}\n"
                    f"Common block_num candidates ({len(common_block_list)}): {common_block_list}\n"
                    f"warp_per_block candidates: {warp_per_block_list}"
                )
                if extra_disp_block_list:
                    print(
                        f"Extra dispatch block_num candidates ({len(extra_disp_block_list)}): {extra_disp_block_list}"
                    )

            best_disp_bw = 0
            best_comb_bw = 0
            best_disp_config = None
            best_comb_config = None

            # Common sweep uses the same block/warp for both dispatch and combine
            # to keep the search space manageable. Asymmetric configs (different
            # block/warp for dispatch vs combine) are not explored here; the extra
            # dispatch sweep below covers over-subscribe block_num candidates.
            if rank == 0:
                print(f"\n{'#' * 60}")
                print("Common sweep (same block/warp for dispatch and combine)")
                print(f"{'#' * 60}")

            for block_num in common_block_list:
                for warp_per_block in warp_per_block_list:
                    if rank == 0:
                        print(f"\n{'=' * 60}")
                        print(f"block_num={block_num}, warp_per_block={warp_per_block}")
                        print(f"{'=' * 60}")

                    disp_bw, comb_bw, _, _ = benchmark.run(
                        op,
                        dispatch_block_num=block_num,
                        dispatch_warp_per_block=warp_per_block,
                        combine_block_num=block_num,
                        combine_warp_per_block=warp_per_block,
                    )

                    if disp_bw > best_disp_bw:
                        best_disp_bw = disp_bw
                        best_disp_config = (block_num, warp_per_block)
                    if comb_bw > best_comb_bw:
                        best_comb_bw = comb_bw
                        best_comb_config = (block_num, warp_per_block)

            # --- Extra dispatch sweep: over-subscribe, fix combine at best ---
            if extra_disp_block_list:
                if rank == 0:
                    print(f"\n{'#' * 60}")
                    print(
                        f"Extra dispatch sweep "
                        f"(combine fixed at block_num={best_comb_config[0]}, wpb={best_comb_config[1]})"
                    )
                    print(f"{'#' * 60}")

                for block_num in extra_disp_block_list:
                    for warp_per_block in warp_per_block_list:
                        if rank == 0:
                            print(f"\n{'=' * 60}")
                            print(
                                f"Dispatch: block_num={block_num}, warp_per_block={warp_per_block}"
                            )
                            print(f"{'=' * 60}")

                        disp_bw, _, _, _ = benchmark.run(
                            op,
                            dispatch_block_num=block_num,
                            dispatch_warp_per_block=warp_per_block,
                            combine_block_num=best_comb_config[0],
                            combine_warp_per_block=best_comb_config[1],
                        )

                        if disp_bw > best_disp_bw:
                            best_disp_bw = disp_bw
                            best_disp_config = (block_num, warp_per_block)

            if rank == 0:
                print(f"\n{'=' * 60}")
                print("Performance Summary:")
                print(f"{'=' * 60}")
                disp_dtype_str = str(data_type).split(".")[-1]
                comb_dtype_str = str(combine_data_type).split(".")[-1]
                print(
                    f"Best Dispatch  ({disp_dtype_str}): {best_disp_bw:.2f} GB/s "
                    f"at block_num={best_disp_config[0]}, warp_per_block={best_disp_config[1]}"
                )
                print(
                    f"Best Combine   ({comb_dtype_str}): {best_comb_bw:.2f} GB/s "
                    f"at block_num={best_comb_config[0]}, warp_per_block={best_comb_config[1]}"
                )
                print(f"{'=' * 60}")

                if save_tuning_config and best_disp_config and best_comb_config:
                    _save_intranode_tuning_result(
                        save_tuning_config,
                        world_size=world_size,
                        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
                        dispatch_hidden_dim=hidden_dim,
                        combine_hidden_dim=combine_hidden_dim,
                        data_type=data_type,
                        combine_data_type=combine_data_type,
                        quant_type=quant_type,
                        best_disp_config=best_disp_config,
                        best_disp_bw=best_disp_bw,
                        best_comb_config=best_comb_config,
                        best_comb_bw=best_comb_bw,
                        zero_copy=bool(zero_copy),
                    )

        else:
            raise ValueError(f"Unknown command: {cmd}")


def bench_dispatch_combine(
    max_num_inp_token_per_rank,
    dtype,
    hidden_dim=7168,
    cmd="bench",
    zero_copy=1,
    quant_type="none",
    dispatch_block_num=None,
    dispatch_warp_per_block=None,
    combine_block_num=None,
    combine_warp_per_block=None,
    world_size=8,
    num_experts_per_rank=32,
    num_experts_per_token=8,
    combine_data_type=None,
    max_total_recv_tokens=0,
    save_tuning_config=None,
):
    if combine_data_type is None:
        combine_data_type = dtype
    port = get_free_port()
    torch.multiprocessing.spawn(
        _bench_dispatch_combine,
        args=(
            world_size,
            port,
            max_num_inp_token_per_rank,
            dtype,
            hidden_dim,
            0,  # scale_dim
            0,  # scale_type_size
            num_experts_per_rank,
            num_experts_per_token,
            cmd,
            zero_copy,
            quant_type,
            dispatch_block_num,
            dispatch_warp_per_block,
            combine_block_num,
            combine_warp_per_block,
            combine_data_type,
            max_total_recv_tokens,
            save_tuning_config,
        ),
        nprocs=world_size,
        join=True,
    )


_DATA_TYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp8_e4m3_fnuz": torch.float8_e4m3fnuz,
    "fp8_e4m3": torch.float8_e4m3fn,
}
if hasattr(torch, "float4_e2m1fn_x2"):
    _DATA_TYPE_MAP["fp4"] = torch.float4_e2m1fn_x2

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark EP Dispatch Combine")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of input tokens per rank (default: 4096)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp8_e4m3_fnuz", "fp8_e4m3", "fp4"],
        help="Data type of dispatch / combine",
    )
    parser.add_argument(
        "--cmd",
        type=str,
        default="bench",
        choices=["bench", "stress", "tuning"],
        help="Available subcommands: bench (single config), stress (stress test), tuning (test multiple configs)",
    )
    parser.add_argument(
        "--zero-copy",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable zero-copy mode: 1 (default, enabled) or 0 (disabled). When enabled, sets use_external_inp_buf=False",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default="none",
        choices=["none", "fp8_direct_cast"],
        help=(
            "Quantization method used inside Combine. "
            "'fp8_direct_cast' is the BF16<->FP8 direct cast path."
        ),
    )
    parser.add_argument(
        "--dispatch-block-num",
        type=int,
        default=None,
        help="Override dispatch block_num for bench/stress. Ignored when --cmd tuning.",
    )
    parser.add_argument(
        "--dispatch-warp-per-block",
        type=int,
        default=None,
        help="Override dispatch warp_per_block for bench/stress. Ignored when --cmd tuning.",
    )
    parser.add_argument(
        "--combine-block-num",
        type=int,
        default=None,
        help="Override combine block_num for bench/stress. Ignored when --cmd tuning.",
    )
    parser.add_argument(
        "--combine-warp-per-block",
        type=int,
        default=None,
        help="Override combine warp_per_block for bench/stress. Ignored when --cmd tuning.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=8,
        help="Number of GPUs (EP degree). Use 4 for EP4, 8 for EP8 (default: 8)",
    )
    parser.add_argument(
        "--num-experts-per-rank",
        type=int,
        default=None,
        help="Number of experts per rank. Defaults to 256 // world_size (e.g. 32 for EP8, 64 for EP4)",
    )
    parser.add_argument(
        "--num-experts-per-token",
        type=int,
        default=8,
        help="Number of experts per token (top-k, default: 8)",
    )
    parser.add_argument(
        "--max-recv-total-tokens",
        type=int,
        default=0,
        help="Maximum total number of received tokens across all ranks (default: 0, meaning no limit)",
    )
    parser.add_argument(
        "--combine-dtype",
        type=str,
        default=None,
        choices=["bf16", "fp8_e4m3_fnuz", "fp8_e4m3"],
        help=(
            "Data type for combine phase (tuning only). "
            "When set, Phase 2 creates a separate op with this dtype. "
        ),
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
    args = parser.parse_args()

    if args.num_experts_per_rank is None:
        args.num_experts_per_rank = 256 // args.world_size

    # Resolve combine dtype: default to same as dispatch
    combine_dtype_str = args.combine_dtype if args.combine_dtype else args.dtype

    dispatch_dtype = _DATA_TYPE_MAP[args.dtype]
    combine_dtype = _DATA_TYPE_MAP[combine_dtype_str]

    base_hidden_dim = args.hidden_dim
    dispatch_hidden_dim = (
        base_hidden_dim // 2 if _is_fp4x2_dtype(dispatch_dtype) else base_hidden_dim
    )

    print(
        f"Running {args.cmd} with max_tokens_per_rank: {args.max_tokens}, "
        f"dispatch_dtype: {args.dtype}, combine_dtype: {combine_dtype_str}, "
        f"hidden_dim: {base_hidden_dim}, "
        f"world_size(EP): {args.world_size}, "
        f"num_experts_per_rank: {args.num_experts_per_rank}, "
        f"num_experts_per_token: {args.num_experts_per_token}, "
        f"zero_copy: {'true' if args.zero_copy else 'false'}, "
        f"quant_type: {args.quant_type}, "
        f"dispatch_block_num: {args.dispatch_block_num}, "
        f"dispatch_warp_per_block: {args.dispatch_warp_per_block}, "
        f"combine_block_num: {args.combine_block_num}, "
        f"combine_warp_per_block: {args.combine_warp_per_block}"
    )
    print("-" * 60)
    bench_dispatch_combine(
        max_num_inp_token_per_rank=args.max_tokens,
        dtype=dispatch_dtype,
        hidden_dim=dispatch_hidden_dim,
        cmd=args.cmd,
        zero_copy=args.zero_copy,
        quant_type=args.quant_type,
        dispatch_block_num=args.dispatch_block_num,
        dispatch_warp_per_block=args.dispatch_warp_per_block,
        combine_block_num=args.combine_block_num,
        combine_warp_per_block=args.combine_warp_per_block,
        world_size=args.world_size,
        num_experts_per_rank=args.num_experts_per_rank,
        num_experts_per_token=args.num_experts_per_token,
        combine_data_type=combine_dtype,
        max_total_recv_tokens=args.max_recv_total_tokens,
        save_tuning_config=args.save_tuning_config,
    )
