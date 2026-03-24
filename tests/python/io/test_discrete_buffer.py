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

"""
Tests for discrete (non-contiguous) buffer optimization in XGMI batch
read/write.  The optimized XGMI backend sorts segments to maximize merging and
uses a scatter/gather GPU kernel for many small discrete segments instead of
issuing N individual hipMemcpy calls.

All tests run across two GPUs (GPU 0 → GPU 1) via XGMI links to exercise the
real cross-device communication path.

Tests cover:
  1. Correctness: strided, shuffled, and random discrete patterns
  2. Performance: batch_write latency benchmark across segment counts and sizes
"""

import os
import pytest
import time
import torch
from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    MemoryDesc,
    XgmiBackendConfig,
    set_log_level,
)

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires at least 2 GPUs"
)

SRC_GPU = 0
DST_GPU = 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xgmi_engine():
    set_log_level("info")
    config = IOEngineConfig(host="", port=0)
    engine = IOEngine(key="discrete_buf_engine", config=config)
    xgmi_config = XgmiBackendConfig(num_streams=64, num_events=64)
    engine.create_backend(BackendType.XGMI, xgmi_config)
    yield engine


def _alloc_pair(engine, src_gpu, dst_gpu, num_elements):
    """Allocate a src tensor filled with random data and a zeroed dst tensor."""
    src = torch.randint(
        1, 256, (num_elements,),
        dtype=torch.uint8,
        device=torch.device("cuda", src_gpu),
    )
    dst = torch.zeros(
        num_elements,
        dtype=torch.uint8,
        device=torch.device("cuda", dst_gpu),
    )
    src_mem = engine.register_torch_tensor(src)
    dst_mem = engine.register_torch_tensor(dst)
    return src, dst, src_mem, dst_mem


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

def test_discrete_batch_write_strided(xgmi_engine):
    """batch_write with strided (non-contiguous) offsets across GPUs."""
    num_segments = 64
    seg_size = 4096
    stride = seg_size * 4
    total = num_segments * stride + seg_size

    src, dst, src_mem, dst_mem = _alloc_pair(xgmi_engine, SRC_GPU, DST_GPU, total)

    offsets = [i * stride for i in range(num_segments)]
    sizes = [seg_size] * num_segments

    sess = xgmi_engine.create_session(src_mem, dst_mem)
    uid = sess.allocate_transfer_uid()
    status = sess.batch_write(offsets, offsets, sizes, uid)
    status.Wait()
    assert status.Succeeded(), f"batch_write failed: {status.Message()}"

    src_cpu, dst_cpu = src.cpu(), dst.cpu()
    for i in range(num_segments):
        off = offsets[i]
        assert torch.equal(src_cpu[off : off + seg_size], dst_cpu[off : off + seg_size]), (
            f"mismatch at segment {i}"
        )


def test_discrete_batch_read_strided(xgmi_engine):
    """batch_read with strided offsets across GPUs."""
    num_segments = 64
    seg_size = 4096
    stride = seg_size * 4
    total = num_segments * stride + seg_size

    src, dst, src_mem, dst_mem = _alloc_pair(xgmi_engine, SRC_GPU, DST_GPU, total)

    offsets = [i * stride for i in range(num_segments)]
    sizes = [seg_size] * num_segments

    sess = xgmi_engine.create_session(dst_mem, src_mem)
    uid = sess.allocate_transfer_uid()
    status = sess.batch_read(offsets, offsets, sizes, uid)
    status.Wait()
    assert status.Succeeded(), f"batch_read failed: {status.Message()}"

    src_cpu, dst_cpu = src.cpu(), dst.cpu()
    for i in range(num_segments):
        off = offsets[i]
        assert torch.equal(src_cpu[off : off + seg_size], dst_cpu[off : off + seg_size])


def test_discrete_batch_shuffled_offsets(xgmi_engine):
    """Segments arrive in random order; sort+merge should still produce correct
    results and merge them optimally when they are logically contiguous."""
    num_segments = 32
    seg_size = 1024
    total = num_segments * seg_size

    src, dst, src_mem, dst_mem = _alloc_pair(xgmi_engine, SRC_GPU, DST_GPU, total)

    indices = list(range(num_segments))
    import random
    random.seed(42)
    random.shuffle(indices)

    offsets = [indices[i] * seg_size for i in range(num_segments)]
    sizes = [seg_size] * num_segments

    sess = xgmi_engine.create_session(src_mem, dst_mem)
    uid = sess.allocate_transfer_uid()
    status = sess.batch_write(offsets, offsets, sizes, uid)
    status.Wait()
    assert status.Succeeded()

    assert torch.equal(src.cpu(), dst.cpu()), (
        "Shuffled-offset batch_write should produce identical result to full copy"
    )


def test_discrete_batch_varying_sizes(xgmi_engine):
    """Segments with different sizes (non-uniform)."""
    seg_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 100, 300, 700]
    num_segments = len(seg_sizes)
    gap = 512
    total = sum(seg_sizes) + gap * num_segments

    src, dst, src_mem, dst_mem = _alloc_pair(xgmi_engine, SRC_GPU, DST_GPU, total)

    src_offsets = []
    dst_offsets = []
    cursor = 0
    for s in seg_sizes:
        src_offsets.append(cursor)
        dst_offsets.append(cursor)
        cursor += s + gap

    sess = xgmi_engine.create_session(src_mem, dst_mem)
    uid = sess.allocate_transfer_uid()
    status = sess.batch_write(src_offsets, dst_offsets, seg_sizes, uid)
    status.Wait()
    assert status.Succeeded()

    src_cpu, dst_cpu = src.cpu(), dst.cpu()
    for i in range(num_segments):
        off = src_offsets[i]
        sz = seg_sizes[i]
        assert torch.equal(src_cpu[off : off + sz], dst_cpu[off : off + sz])


def test_discrete_batch_engine_api(xgmi_engine):
    """Test discrete batch via the IOEngine (non-session) API."""
    num_segments = 64
    seg_size = 2048
    stride = seg_size * 2
    total = num_segments * stride + seg_size

    src, dst, src_mem, dst_mem = _alloc_pair(xgmi_engine, SRC_GPU, DST_GPU, total)

    offsets = [i * stride for i in range(num_segments)]
    sizes = [seg_size] * num_segments

    uid = xgmi_engine.allocate_transfer_uid()
    statuses = xgmi_engine.batch_write(
        [src_mem], [offsets], [dst_mem], [offsets], [sizes], [uid]
    )
    statuses[0].Wait()
    assert statuses[0].Succeeded()

    src_cpu, dst_cpu = src.cpu(), dst.cpu()
    for i in range(num_segments):
        off = offsets[i]
        assert torch.equal(src_cpu[off : off + seg_size], dst_cpu[off : off + seg_size])


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------

TEST_CASES = [
    (32, 4096, "32 segs x 4KB"),
    (64, 4096, "64 segs x 4KB"),
    (128, 4096, "128 segs x 4KB"),
    (256, 4096, "256 segs x 4KB"),
    (512, 2048, "512 segs x 2KB"),
    (1024, 1024, "1024 segs x 1KB"),
    (128, 16384, "128 segs x 16KB"),
    (64, 65536, "64 segs x 64KB"),
]

WARMUP = 10
ITERATIONS = 100


def _run_batch_perf(xgmi_engine, op):
    """Shared benchmark logic for batch_write / batch_read."""
    print(f"\n  GPU {SRC_GPU} -> GPU {DST_GPU} {op} benchmark")
    print("  %-25s %12s" % ("TestCase", f"{op}(us)"))
    print("  " + "-" * 40)

    for num_segments, seg_size, desc in TEST_CASES:
        stride = seg_size * 4
        total = num_segments * stride + seg_size

        src, dst, src_mem, dst_mem = _alloc_pair(xgmi_engine, SRC_GPU, DST_GPU, total)

        offsets = [i * stride for i in range(num_segments)]
        sizes_list = [seg_size] * num_segments

        if op == "batch_write":
            sess = xgmi_engine.create_session(src_mem, dst_mem)
        else:
            sess = xgmi_engine.create_session(dst_mem, src_mem)

        batch_fn = getattr(sess, op)

        for _ in range(WARMUP):
            uid = sess.allocate_transfer_uid()
            s = batch_fn(offsets, offsets, sizes_list, uid)
            s.Wait()

        t0 = time.perf_counter()
        for _ in range(ITERATIONS):
            uid = sess.allocate_transfer_uid()
            s = batch_fn(offsets, offsets, sizes_list, uid)
            s.Wait()
        avg_us = (time.perf_counter() - t0) / ITERATIONS * 1e6

        print("  %-25s %12.1f" % (desc, avg_us))

        xgmi_engine.deregister_memory(src_mem)
        xgmi_engine.deregister_memory(dst_mem)


@pytest.mark.skipif(
    not os.environ.get("MORI_RUN_PERF_TESTS"), reason="set MORI_RUN_PERF_TESTS=1 to run benchmarks"
)
def test_discrete_batch_write_performance(xgmi_engine):
    """Benchmark batch_write across GPUs with various discrete segment patterns."""
    _run_batch_perf(xgmi_engine, "batch_write")


@pytest.mark.skipif(
    not os.environ.get("MORI_RUN_PERF_TESTS"), reason="set MORI_RUN_PERF_TESTS=1 to run benchmarks"
)
def test_discrete_batch_read_performance(xgmi_engine):
    """Benchmark batch_read across GPUs with various discrete segment patterns."""
    _run_batch_perf(xgmi_engine, "batch_read")


# ---------------------------------------------------------------------------
# Multi-process tests (two processes, each binding one GPU)
# Follows the same pattern as benchmark.py xgmi_multiprocess and
# TorchDistProcessManager in tests/python/utils.py.
# ---------------------------------------------------------------------------

import torch.distributed as dist
import torch.multiprocessing as mp
from tests.python.utils import TorchDistContext, get_free_port


def _send_bytes(data: bytes, dst: int):
    """Send a byte buffer to a peer rank via torch.distributed."""
    t = torch.ByteTensor(list(data))
    dist.send(torch.tensor([len(data)], dtype=torch.long), dst=dst)
    dist.send(t, dst=dst)


def _recv_bytes(src: int) -> bytes:
    """Receive a byte buffer from a peer rank via torch.distributed."""
    n = torch.zeros(1, dtype=torch.long)
    dist.recv(n, src=src)
    buf = torch.zeros(int(n.item()), dtype=torch.uint8)
    dist.recv(buf, src=src)
    return bytes(buf.tolist())


def _mp_discrete_worker(rank, world_size, master_port, result_queue):
    """Worker spawned by torch.multiprocessing.Process (uses 'spawn' start
    method so CUDA context is fresh).

    rank 0 = initiator on GPU 0, rank 1 = target on GPU 1.
    """
    try:
        with TorchDistContext(
            rank=rank,
            world_size=world_size,
            master_addr="localhost",
            master_port=str(master_port),
            device_id=rank,
            backend="gloo",
        ):
            set_log_level("info")
            gpu_id = rank

            config = IOEngineConfig(host="", port=0)
            engine = IOEngine(key=f"xgmi_mp_{rank}", config=config)
            xgmi_config = XgmiBackendConfig(num_streams=64, num_events=64)
            engine.create_backend(BackendType.XGMI, xgmi_config)

            engine_desc = engine.get_engine_desc()
            desc_bytes = engine_desc.pack()

            peer_rank = 1 - rank
            if rank == 0:
                peer_desc_bytes = _recv_bytes(src=peer_rank)
                peer_desc = EngineDesc.unpack(peer_desc_bytes)
                engine.register_remote_engine(peer_desc)
                _send_bytes(desc_bytes, dst=peer_rank)
            else:
                _send_bytes(desc_bytes, dst=peer_rank)
                peer_desc_bytes = _recv_bytes(src=peer_rank)
                peer_desc = EngineDesc.unpack(peer_desc_bytes)
                engine.register_remote_engine(peer_desc)

            num_segments = 128
            seg_size = 4096
            stride = seg_size * 4
            total = num_segments * stride + seg_size

            tensor = torch.randint(
                1, 256, (total,), dtype=torch.uint8,
                device=torch.device("cuda", gpu_id),
            )
            mem = engine.register_torch_tensor(tensor)
            mem_bytes = mem.pack()

            if rank == 0:
                peer_mem_bytes = _recv_bytes(src=peer_rank)
                remote_mem = MemoryDesc.unpack(peer_mem_bytes)
                _send_bytes(mem_bytes, dst=peer_rank)
            else:
                _send_bytes(mem_bytes, dst=peer_rank)
                peer_mem_bytes = _recv_bytes(src=peer_rank)
                remote_mem = MemoryDesc.unpack(peer_mem_bytes)

            offsets = [i * stride for i in range(num_segments)]
            sizes = [seg_size] * num_segments

            if rank == 0:
                # batch_write: GPU 0 -> GPU 1
                sess = engine.create_session(mem, remote_mem)
                uid = sess.allocate_transfer_uid()
                status = sess.batch_write(offsets, offsets, sizes, uid)
                status.Wait()
                assert status.Succeeded(), f"MP batch_write: {status.Message()}"

            dist.barrier()

            if rank == 0:
                # batch_read: read back from GPU 1 -> GPU 0
                read_tensor = torch.zeros(
                    total, dtype=torch.uint8,
                    device=torch.device("cuda", gpu_id),
                )
                read_mem = engine.register_torch_tensor(read_tensor)
                sess2 = engine.create_session(read_mem, remote_mem)
                uid2 = sess2.allocate_transfer_uid()
                status2 = sess2.batch_read(offsets, offsets, sizes, uid2)
                status2.Wait()
                assert status2.Succeeded(), f"MP batch_read: {status2.Message()}"

                src_cpu = tensor.cpu()
                read_cpu = read_tensor.cpu()
                for i in range(num_segments):
                    off = offsets[i]
                    assert torch.equal(
                        src_cpu[off : off + seg_size],
                        read_cpu[off : off + seg_size],
                    ), f"MP read-back mismatch at segment {i}"

            dist.barrier()
            result_queue.put(("PASS", ""))

    except Exception as e:
        import traceback
        result_queue.put(("FAIL", f"{e}\n{traceback.format_exc()}"))


def test_discrete_batch_multiprocess():
    """Two-process XGMI test: process 0 (GPU 0) and process 1 (GPU 1) each
    create their own IOEngine and exchange MemoryDesc via torch.distributed.
    Initiator does batch_write then batch_read with discrete offsets."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 GPUs")

    master_port = get_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(2):
        p = ctx.Process(target=_mp_discrete_worker, args=(rank, 2, master_port, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=120)
        if p.is_alive():
            p.terminate()
            p.join()
            pytest.fail(f"Multi-process test timed out: worker {p.pid} did not finish within 120s")
        assert p.exitcode == 0, f"Worker process {p.pid} exited with code {p.exitcode}"

    world_size = 2
    results = []
    for _ in range(world_size):
        results.append(result_queue.get(timeout=10))

    for status, msg in results:
        assert status == "PASS", f"Multi-process test failed: {msg}"


# ---------------------------------------------------------------------------
# Multi-process performance benchmark
# ---------------------------------------------------------------------------

MP_TEST_CASES = [
    (32, 4096, "32 segs x 4KB"),
    (64, 4096, "64 segs x 4KB"),
    (128, 4096, "128 segs x 4KB"),
    (256, 4096, "256 segs x 4KB"),
    (512, 2048, "512 segs x 2KB"),
    (1024, 1024, "1024 segs x 1KB"),
    (128, 16384, "128 segs x 16KB"),
    (64, 65536, "64 segs x 64KB"),
]

MP_WARMUP = 10
MP_ITERATIONS = 100


def _mp_perf_worker(rank, world_size, master_port, result_queue):
    """Multi-process performance benchmark worker."""
    try:
        with TorchDistContext(
            rank=rank,
            world_size=world_size,
            master_addr="localhost",
            master_port=str(master_port),
            device_id=rank,
            backend="gloo",
        ):
            set_log_level("warning")
            gpu_id = rank

            config = IOEngineConfig(host="", port=0)
            engine = IOEngine(key=f"xgmi_perf_{rank}", config=config)
            xgmi_config = XgmiBackendConfig(num_streams=64, num_events=64)
            engine.create_backend(BackendType.XGMI, xgmi_config)

            engine_desc = engine.get_engine_desc()
            desc_bytes = engine_desc.pack()

            peer_rank = 1 - rank
            if rank == 0:
                peer_desc_bytes = _recv_bytes(src=peer_rank)
                peer_desc = EngineDesc.unpack(peer_desc_bytes)
                engine.register_remote_engine(peer_desc)
                _send_bytes(desc_bytes, dst=peer_rank)
            else:
                _send_bytes(desc_bytes, dst=peer_rank)
                peer_desc_bytes = _recv_bytes(src=peer_rank)
                peer_desc = EngineDesc.unpack(peer_desc_bytes)
                engine.register_remote_engine(peer_desc)

            perf_lines = []

            for op in ("batch_write", "batch_read"):
                perf_lines.append(
                    f"\n  GPU 0 -> GPU 1 multi-process {op} benchmark"
                )
                perf_lines.append("  %-25s %12s" % ("TestCase", f"{op}(us)"))
                perf_lines.append("  " + "-" * 40)

                for num_segments, seg_size, desc in MP_TEST_CASES:
                    stride = seg_size * 4
                    total = num_segments * stride + seg_size

                    tensor = torch.randint(
                        1, 256, (total,), dtype=torch.uint8,
                        device=torch.device("cuda", gpu_id),
                    )
                    mem = engine.register_torch_tensor(tensor)
                    mem_bytes = mem.pack()

                    if rank == 0:
                        peer_mem_bytes = _recv_bytes(src=peer_rank)
                        remote_mem = MemoryDesc.unpack(peer_mem_bytes)
                        _send_bytes(mem_bytes, dst=peer_rank)
                    else:
                        _send_bytes(mem_bytes, dst=peer_rank)
                        peer_mem_bytes = _recv_bytes(src=peer_rank)
                        remote_mem = MemoryDesc.unpack(peer_mem_bytes)

                    offsets = [i * stride for i in range(num_segments)]
                    sizes_list = [seg_size] * num_segments

                    if rank == 0:
                        if op == "batch_write":
                            sess = engine.create_session(mem, remote_mem)
                        else:
                            sess = engine.create_session(mem, remote_mem)

                        batch_fn = getattr(sess, op)

                        for _ in range(MP_WARMUP):
                            uid = sess.allocate_transfer_uid()
                            s = batch_fn(offsets, offsets, sizes_list, uid)
                            s.Wait()

                        t0 = time.perf_counter()
                        for _ in range(MP_ITERATIONS):
                            uid = sess.allocate_transfer_uid()
                            s = batch_fn(offsets, offsets, sizes_list, uid)
                            s.Wait()
                        avg_us = (time.perf_counter() - t0) / MP_ITERATIONS * 1e6

                        perf_lines.append("  %-25s %12.1f" % (desc, avg_us))

                    dist.barrier()
                    engine.deregister_memory(mem)

            if rank == 0:
                result_queue.put(("PASS", "\n".join(perf_lines)))
            else:
                result_queue.put(("PASS", ""))

    except Exception as e:
        import traceback
        result_queue.put(("FAIL", f"{e}\n{traceback.format_exc()}"))


@pytest.mark.skipif(
    not os.environ.get("MORI_RUN_PERF_TESTS"), reason="set MORI_RUN_PERF_TESTS=1 to run benchmarks"
)
def test_discrete_batch_multiprocess_performance():
    """Multi-process XGMI performance benchmark for discrete buffer
    batch_write and batch_read."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires at least 2 GPUs")

    master_port = get_free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(2):
        p = ctx.Process(target=_mp_perf_worker, args=(rank, 2, master_port, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=300)
        if p.is_alive():
            p.terminate()
            p.join()
            pytest.fail(f"Multi-process perf test timed out: worker {p.pid} did not finish within 300s")
        assert p.exitcode == 0, f"Worker process {p.pid} exited with code {p.exitcode}"

    world_size = 2
    results = []
    for _ in range(world_size):
        results.append(result_queue.get(timeout=10))

    for status, msg in results:
        assert status == "PASS", f"Multi-process perf test failed: {msg}"
        if msg:
            print(msg)
