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

    def sync(self):
        torch.cuda.synchronize()
        dist.barrier()

    def run_once(self, op, test_data):
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

        src_token_pos = op.get_dispatch_src_token_pos()
        # for i, pos in enumerate(src_token_pos):
        #     src_rank = int(pos) // self.config.max_num_inp_token_per_rank
        #     src_id = int(pos) % self.config.max_num_inp_token_per_rank
        #     assert torch.equal(all_rank_input[src_rank][src_id], dispatch_output[i])

        total_recv_num_token = dispatch_recv_num_token[0].item()

        combine_input = op.get_registered_input_buffer(self.config.data_type)
        combine_input[:total_recv_num_token, :].copy_(
            dispatch_output[:total_recv_num_token, :]
        )

        self.sync()
        start_event.record()
        combine_output = op.combine(
            combine_input,
            dispatch_weights,
            dispatch_indicies,
            call_reset=False,
            block_num=80,
            warp_per_block=8,
        )
        end_event.record()
        self.sync()
        comb_duration = start_event.elapsed_time(end_event)

        # for i in range(int(all_rank_num_token[self.config.rank].item())):
        #     pes = [
        #         (idx // self.config.num_experts_per_rank)
        #         for idx in all_rank_indicies[self.config.rank][i].cpu().tolist()
        #     ]
        #     unique_pes = len(set(pes))
        #     got, expected = combine_output[i], all_rank_input[self.config.rank][i] * unique_pes
        #     assert torch.allclose(
        #         got.float(), expected.float(), atol=1e-2, rtol=1e-2
        #     )
        op.reset()
        self.sync()

        element_size = all_rank_input[self.config.rank].element_size()
        total_bytes = total_recv_num_token * self.config.hidden_dim * element_size
        disp_bandwidth = total_bytes / (1024**3) / (disp_duration / (10**3))
        comb_bandwidth = total_bytes / (1024**3) / (comb_duration / (10**3))

        return disp_duration, comb_duration, disp_bandwidth, comb_bandwidth, total_bytes

    def run(self, op, warmup=1, iters=10, always_new_data=True):
        test_data = self.gen_test_data()
        for _ in range(warmup):
            self.run_once(op, test_data)

        disp_duration_us_list = []
        disp_bandwidth_GB_list = []
        comb_duration_us_list = []
        comb_bandwidth_GB_list = []

        # gen test data at each round to eliminate the effect of caching
        if always_new_data:
            test_data_list = [self.gen_test_data() for i in range(iters)]
            print("Is new data !!!!")
        else:
            test_data_list = [test_data for i in range(iters)]

        for i in range(iters):
            self.sync()
            disp_dur, comb_dur, disp_bw, comb_bw, total_bytes = self.run_once(
                op, test_data_list[i]
            )

            disp_dur_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            disp_bw_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            comb_dur_list = [torch.zeros(1) for _ in range(self.config.world_size)]
            comb_bw_list = [torch.zeros(1) for _ in range(self.config.world_size)]

            dist.all_gather(disp_dur_list, torch.tensor([disp_dur * 1000]))
            dist.all_gather(disp_bw_list, torch.tensor([disp_bw]))
            dist.all_gather(comb_dur_list, torch.tensor([comb_dur * 1000]))
            dist.all_gather(comb_bw_list, torch.tensor([comb_bw]))

            disp_duration_us_list.append([int(t.item()) for t in disp_dur_list])
            disp_bandwidth_GB_list.append([int(t.item()) for t in disp_bw_list])
            comb_duration_us_list.append([int(t.item()) for t in comb_dur_list])
            comb_bandwidth_GB_list.append([int(t.item()) for t in comb_bw_list])

        total_bytes_list = [torch.zeros(1) for _ in range(self.config.world_size)]
        dist.all_gather(total_bytes_list, torch.tensor([total_bytes / (1024**2)]))
        total_bytes_list = [t.item() for t in total_bytes_list]

        if self.config.rank == 0:
            for i, duration_us in enumerate(disp_duration_us_list):
                print(
                    f"Round {i} dispatch duration {duration_us} bandwidth {disp_bandwidth_GB_list[i]} avg {sum(disp_bandwidth_GB_list[i]) / self.config.world_size}"
                )
            for i, duration_us in enumerate(comb_duration_us_list):
                print(
                    f"Round {i} combine duration {duration_us} bandwidth {comb_bandwidth_GB_list[i]} avg {sum(comb_bandwidth_GB_list[i]) / self.config.world_size}"
                )

            print(f"Total bytes is {total_bytes_list} MB")


def _bench_dispatch_combine(
    rank,
    world_size,
    port,
    data_type=torch.bfloat16,
    hidden_dim=7168,
    scale_dim=0,
    scale_type_size=0,
    max_num_inp_token_per_rank=4096,
    num_experts_per_rank=32,
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
        warp_num_per_block=4,
        block_num=80,
        use_external_inp_buf=False,
    )
    benchmark = EpDispatchCombineBenchmark(config)

    with TorchDistContext(rank=rank, world_size=world_size, master_port=port) as ctx:
        mori.shmem.shmem_torch_process_group_init("default")
        op = mori.ops.EpDispatchCombineOp(config)
        benchmark.run(op, always_new_data=True)
        # benchmark.output()
        # mori.shmem.shmem_finalize()


def bench_dispatch_combine():
    world_size = 8
    port = get_free_port()
    torch.multiprocessing.spawn(
        _bench_dispatch_combine, args=(world_size, port), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    bench_dispatch_combine()
