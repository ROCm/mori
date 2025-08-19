import pytest
import mori
from tests.python.ops.test_dispatch_combine import EpDispatchCombineTestCase
from tests.python.utils import TorchDistContext, get_free_port, TorchDistProcessManager
import torch
import torch.distributed as dist
import time


class EpDispatchCombineBenchmark(EpDispatchCombineTestCase):
    def __init__(self, config):
        super().__init__(config)

    def gen_test_data(self):
        return super().gen_test_data(use_max_token_num=True)

    def run_once(self, op, test_data, check_result):
        (
            all_rank_num_token,
            all_rank_indicies,
            all_rank_input,
            all_rank_weights,
            all_rank_scales,
        ) = test_data

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        self.sync()
        start_event.record()
        (
            dispatch_output,
            dispatch_weights,
            dispatch_scales,
            dispatch_indicies,
            dispatch_recv_num_token,
        ) = op.dispatch(
            all_rank_input[self.config.rank],
            all_rank_weights[self.config.rank],
            all_rank_scales[self.config.rank],
            all_rank_indicies[self.config.rank],
            block_num=80,
            warp_per_block=16,
        )
        end_event.record()
        self.sync()
        disp_duration = start_event.elapsed_time(end_event)

        if check_result:
            self.check_dispatch_result(
                op,
                test_data,
                dispatch_output,
                dispatch_weights,
                dispatch_scales,
                dispatch_indicies,
                dispatch_recv_num_token,
            )

        total_recv_num_token = dispatch_recv_num_token[0].item()

        combine_input = op.get_registered_input_buffer(self.config.data_type)
        combine_input[:total_recv_num_token, :].copy_(
            dispatch_output[:total_recv_num_token, :]
        )

        self.sync()
        start_event.record()
        combine_output, _ = op.combine(
            combine_input,
            # dispatch_weights,
            None,
            dispatch_indicies,
            call_reset=False,
            block_num=80,
            warp_per_block=8,
        )
        end_event.record()
        self.sync()
        comb_duration = start_event.elapsed_time(end_event)

        if check_result:
            self.check_combine_result(op, test_data, combine_output)
        op.reset()
        self.sync()

        element_size = all_rank_input[self.config.rank].element_size()
        total_bytes = total_recv_num_token * self.config.hidden_dim * element_size
        disp_bandwidth = total_bytes / (1000**3) / (disp_duration / (10**3))
        comb_bandwidth = total_bytes / (1000**3) / (comb_duration / (10**3))

        return disp_duration, comb_duration, disp_bandwidth, comb_bandwidth, total_bytes

    def run(self, op, warmup=1, iters=10):
        test_data = self.gen_test_data()
        for _ in range(warmup):
            self.run_once(op, test_data, True)

        disp_duration_us_list = []
        disp_bandwidth_GB_list = []
        comb_duration_us_list = []
        comb_bandwidth_GB_list = []
        avg_total_bytes_MB_list = []

        test_data_list = [self.gen_test_data() for i in range(iters)]

        for i in range(iters):
            self.sync()
            disp_dur, comb_dur, disp_bw, comb_bw, total_bytes = self.run_once(
                op, test_data_list[i], False
            )

            disp_dur_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            disp_bw_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            comb_dur_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            comb_bw_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            total_bytes_list = [torch.zeros(1) for _ in range(self.config.world_size)]

            dist.all_gather(disp_dur_list, torch.tensor([disp_dur * 1000]))
            dist.all_gather(disp_bw_list, torch.tensor([disp_bw]))
            dist.all_gather(comb_dur_list, torch.tensor([comb_dur * 1000]))
            dist.all_gather(comb_bw_list, torch.tensor([comb_bw]))
            dist.all_gather(total_bytes_list, torch.tensor([total_bytes / (1024**2)]))

            disp_duration_us_list.append([int(t.item()) for t in disp_dur_list])
            disp_bandwidth_GB_list.append([int(t.item()) for t in disp_bw_list])
            comb_duration_us_list.append([int(t.item()) for t in comb_dur_list])
            comb_bandwidth_GB_list.append([int(t.item()) for t in comb_bw_list])
            avg_total_bytes_MB_list.append(
                int(torch.tensor(total_bytes_list).mean().item())
            )

        theoretical_peak_bw = 50 * self.config.world_size
        if self.config.rank == 0:
            print("Dispatch result:")
            for i, duration_us in enumerate(disp_duration_us_list):
                algo_bw = sum(disp_bandwidth_GB_list[i]) / self.config.world_size
                bus_bw = int(
                    algo_bw * (self.config.world_size - 1) / self.config.world_size
                )
                print(
                    f"Round {i} duration(us) {duration_us} "
                    f"bandwidth(GB/s) {disp_bandwidth_GB_list[i]} "
                    f"avg bytes(MB) {avg_total_bytes_MB_list[i]} bw {algo_bw}({theoretical_peak_bw})"
                )

            print()
            print("Combine result:")
            for i, duration_us in enumerate(comb_duration_us_list):
                algo_bw = sum(comb_bandwidth_GB_list[i]) / self.config.world_size
                bus_bw = int(
                    algo_bw * (self.config.world_size - 1) / self.config.world_size
                )
                print(
                    f"Round {i} duration(us) {duration_us} "
                    f"bandwidth(GB/s) {comb_bandwidth_GB_list[i]} "
                    f"avg bytes(MB) {avg_total_bytes_MB_list[i]} bw {algo_bw}({theoretical_peak_bw})"
                )


def _bench_dispatch_combine(
    rank,
    world_size,
    port,
    max_num_inp_token_per_rank=4096,
    data_type=torch.bfloat16,
    hidden_dim=4096,
    scale_dim=0,
    scale_type_size=0,
    num_experts_per_rank=16,
    num_experts_per_token=8,
):
    config = mori.ops.EpDispatchCombineConfig(
        data_type=data_type,
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=scale_type_size,
        max_token_type_size=2,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_experts_per_rank,
        num_experts_per_token=num_experts_per_token,
        warp_num_per_block=16,
        block_num=80,
        use_external_inp_buf=False,
    )
    benchmark = EpDispatchCombineBenchmark(config)

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port) as ctx:
        mori.shmem.shmem_torch_process_group_init("default")
        op = mori.ops.EpDispatchCombineOp(config)
        benchmark.run(op)
        # benchmark.output()
        # mori.shmem.shmem_finalize()


def bench_dispatch_combine(max_num_inp_token_per_rank=4096):
    world_size = 8
    port = get_free_port()
    torch.multiprocessing.spawn(
        _bench_dispatch_combine,
        args=(world_size, port, max_num_inp_token_per_rank),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark EP Dispatch Combine")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of input tokens per rank (default: 4096)",
    )

    args = parser.parse_args()

    print(f"Running benchmark with max_tokens_per_rank: {args.max_tokens}")
    print("-" * 60)
    bench_dispatch_combine(max_num_inp_token_per_rank=args.max_tokens)
