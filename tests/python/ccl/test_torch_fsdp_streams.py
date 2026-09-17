"""Exercise FSDP buffer reuse and completion on separate GPU streams."""

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from mori.ccl.torch_fsdp import MoriSdmaAllGather


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
                    if comm.layout is not None:
                        comm.layout.prepare_output(
                            splits, count, world_size, torch.float32, device,
                            [[torch.float32], [torch.float32]],
                            [[size] for size in splits], True,
                        )
                gather.wait_stream(producer)
                with torch.cuda.stream(gather), patch.object(
                    collective, enqueue_name, side_effect=delayed_enqueue
                ):
                    work = comm(output, source, dist.group.WORLD, async_op=async_op)
                    done = gather.record_event()
                # Releasing buffers while SDMA is pending must not allow reuse.
                del source
                comm._clear_output()
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
