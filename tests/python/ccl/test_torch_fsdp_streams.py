"""Exercise FSDP buffer reuse and completion on separate GPU streams."""

import gc
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._all_gather_layout import AllGatherInputMetadata

from mori.ccl.torch_fsdp import MoriSdmaAllGather, MoriSdmaAllGatherPool


def _standalone_output_allocation_case(rank: int, world_size: int) -> None:
    device = torch.device("cuda", rank)
    outer_pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(outer_pool, device=device):
        # Force ordinary small allocations away from their IPC allocation base.
        prefix = torch.full((256,), -99.0, device=device)
        for zero_copy in (False, True):
            comm = MoriSdmaAllGather(zero_copy_output=zero_copy)
            retained = []
            for count in (4096, 8192):
                output = comm.allocate(
                    (count * world_size,), dtype=torch.float32, device=torch.device("cuda")
                )
                segment = next(
                    segment for segment in torch.cuda.memory_snapshot()
                    if segment["address"] <= output.data_ptr()
                    < segment["address"] + segment["total_size"]
                )
                assert output.data_ptr() == segment["address"]
                reused = comm.allocate(output.shape, dtype=output.dtype, device=device)
                assert reused.data_ptr() == output.data_ptr()
                splits = [count // 4, 3 * count // 4]
                comm.layout.prepare_output(AllGatherInputMetadata(
                    input_split_sizes=splits, input_numel=count, world_size=world_size,
                    dtype=torch.float32, device=device,
                    can_use_param_contiguous_output=True,
                ))
                source = torch.arange(count, device=device, dtype=torch.float32) + rank * count
                comm(output, source, dist.group.WORLD, async_op=True).wait()
                peer_inputs = [
                    (torch.arange(count, device=device, dtype=torch.float32) + peer * count)
                    .split(splits if zero_copy else [count])
                    for peer in range(world_size)
                ]
                expected = torch.cat([
                    peer_inputs[peer][i]
                    for i in range(len(peer_inputs[0])) for peer in range(world_size)
                ])
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                torch.testing.assert_close(prefix, torch.full_like(prefix, -99.0))
                comm.release_output()
                # Growing an output must remain safe while old parameter views survive.
                retained.append(output)
            comm._deregister_output_buffer_if_needed()


def _metadata_cache_case(rank: int, world_size: int) -> None:
    device = torch.device("cuda", rank)
    count = 4096
    pool = MoriSdmaAllGatherPool([count * world_size * 4], group=dist.group.WORLD, device=device)
    comm = MoriSdmaAllGather(output_pool=pool)
    pool.initialize()
    producers = [torch.cuda.Stream(), torch.cuda.Stream()]
    gathers = [torch.cuda.Stream(), torch.cuda.Stream()]
    consumer = torch.cuda.Stream()

    def prepare(splits, requested_device=device):
        return comm.layout.prepare_output(AllGatherInputMetadata(
            input_split_sizes=splits, input_numel=count, world_size=world_size,
            dtype=torch.float32, device=requested_device,
            can_use_param_contiguous_output=True,
        ))

    first = prepare([1024, 3072], torch.device("cuda"))
    with patch.object(torch, "tensor", side_effect=AssertionError("cache miss")):
        assert prepare([1024, 3072])[0] is first[0]
    with torch.cuda.device((rank + 1) % world_size):
        other = prepare([1024, 3072], torch.device("cuda"))
        assert other[0].device.index == (rank + 1) % world_size
    del first, other

    for step in range(4):
        producer, gather = producers[step % 2], gathers[step % 2]
        splits = [1024, 3072] if step % 2 == 0 else [2048, 2048]
        with torch.cuda.stream(producer):
            metadata = prepare(splits)
            with patch.object(torch, "tensor", side_effect=AssertionError("cache miss")):
                hit = prepare(splits)
            assert hit[0] is metadata[0] and hit[1] is metadata[1]
            output = comm.allocate((count * world_size,), dtype=torch.float32, device=device)
            source = torch.arange(count, device=device, dtype=torch.float32) + rank * count
        gather.wait_stream(producer)
        with torch.cuda.stream(gather):
            torch.cuda._sleep(20_000_000)
            work = comm(output, source, dist.group.WORLD, async_op=True)
        del metadata, hit, source
        # Evict metadata while the preceding raw-pointer kernel is still pending.
        with torch.cuda.stream(producers[(step + 1) % 2]):
            prepare([512, 3584])
            comm._clear_prepared_output()
            churn = [torch.full((2,), -999, device=device, dtype=torch.int64) for _ in range(32)]
        with torch.cuda.stream(consumer):
            work.wait()
            result = output.clone()
            comm.release_output()
        torch.cuda.synchronize()
        peer_inputs = [
            (torch.arange(count, device=device, dtype=torch.float32) + peer * count).split(splits)
            for peer in range(world_size)
        ]
        expected = torch.cat([peer_inputs[peer][i] for i in range(2) for peer in range(world_size)])
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        del churn
    pool.close()


def _worker(rank: int, world_size: int, port: int) -> None:
    import mori.shmem as shmem

    try:
        utils = importlib.import_module("tests.python.utils")
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        utils = importlib.import_module("tests.python.utils")

    with utils.TorchDistContext(rank=rank, world_size=world_size, master_port=port):
        shmem.shmem_torch_process_group_init("default")
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        _standalone_output_allocation_case(rank, world_size)
        gc.collect()
        dist.barrier()
        _metadata_cache_case(rank, world_size)
        producer = torch.cuda.Stream()
        gather = torch.cuda.Stream()
        consumers = [torch.cuda.Stream(), torch.cuda.Stream()]
        count = 1024 * 1024
        splits = [count // 4, 3 * count // 4]

        for zero_copy in (False, True):
            comm = MoriSdmaAllGather(zero_copy_output=zero_copy)
            collective = comm._get_collective(dist.group.WORLD)
            output = comm.allocate(
                (count * world_size,), dtype=torch.float32, device=device
            )
            enqueue_name = "enqueue_param_contiguous" if zero_copy else "enqueue"
            enqueue = getattr(collective, enqueue_name)

            def delayed_enqueue(*args, **kwargs):
                torch.cuda._sleep(200_000_000)
                return enqueue(*args, **kwargs)

            for async_op in (False, True):
                expected_parts = [
                    torch.full((size,), float(peer + 1), device=device)
                    for size in (splits if zero_copy else [count])
                    for peer in range(world_size)
                ]
                expected = torch.cat(expected_parts)
                torch.cuda.synchronize()
                with torch.cuda.stream(producer):
                    source = torch.full((count,), float(rank + 1), device=device)
                    comm.layout.prepare_output(
                        AllGatherInputMetadata(
                            input_split_sizes=splits, input_numel=count,
                            world_size=world_size, dtype=torch.float32, device=device,
                            can_use_param_contiguous_output=True,
                        )
                    )
                gather.wait_stream(producer)
                with torch.cuda.stream(gather), patch.object(
                    collective, enqueue_name, side_effect=delayed_enqueue
                ):
                    work = comm(output, source, dist.group.WORLD, async_op=async_op)
                    done = gather.record_event()
                # Releasing buffers while SDMA is pending must not allow reuse.
                del source
                comm._clear_prepared_output()
                with torch.cuda.stream(producer):
                    replacement = torch.full((count,), -99.0, device=device)
                results = []
                for consumer in consumers:
                    with torch.cuda.stream(consumer):
                        if async_op:
                            if not isinstance(work, dist.Work):
                                raise AssertionError("async all-gather must return Work")
                            work.wait()
                        else:
                            if work is not None:
                                raise AssertionError("sync all-gather must use stream order")
                            consumer.wait_event(done)
                        results.append(output.clone())
                torch.cuda.synchronize()
                for result in results:
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                del replacement
            comm._deregister_output_buffer_if_needed()
        dist.barrier()
        shmem.shmem_finalize()


@pytest.mark.skipif(
    os.environ.get("MORI_ENABLE_SDMA", "").lower() not in ("1", "true", "yes", "on")
    or torch.cuda.device_count() < 2,
    reason="requires MORI_ENABLE_SDMA=1 and two GPUs",
)
def test_all_gather_stream_lifetimes() -> None:
    try:
        utils = importlib.import_module("tests.python.utils")
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        utils = importlib.import_module("tests.python.utils")
    torch.multiprocessing.spawn(_worker, args=(2, utils.get_free_port()), nprocs=2)
