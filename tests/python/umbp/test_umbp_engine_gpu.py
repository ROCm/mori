"""Real-device checks for the torch-facing engine KV allocator."""

import inspect
import os

import pytest


pytestmark = pytest.mark.gpu


def _require_gpu():
    if not os.path.exists("/dev/kfd"):
        pytest.skip("/dev/kfd is not available")
    torch = pytest.importorskip("torch")
    if torch.version.hip is None or torch.cuda.device_count() == 0:
        pytest.skip("torch has no visible HIP device")
    if "no_split" not in inspect.signature(torch.cuda.MemPool).parameters:
        pytest.skip("this torch build has no MemPool(no_split=...) support")
    return torch


def test_no_split_mempool_is_cached_per_device_and_tensor_starts_at_vmm_base():
    torch = _require_gpu()
    from mori.umbp.engine import allocator_stats, kv_mem_pool, lookup_allocation

    torch.cuda.set_device(0)
    first_pool = kv_mem_pool(0)
    assert first_pool is kv_mem_pool("cuda:0")
    before = allocator_stats()["allocations"]
    with torch.cuda.use_mem_pool(first_pool):
        tensor = torch.empty(8 << 20, dtype=torch.uint8, device="cuda:0")
    tensor.copy_(torch.arange(tensor.numel(), device="cuda:0", dtype=torch.int64) % 251)
    torch.cuda.synchronize()

    allocation = lookup_allocation(tensor.data_ptr(), tensor.nbytes)
    assert allocation is not None
    assert allocation["base"] == tensor.data_ptr()
    assert allocation["requested_size"] >= tensor.nbytes
    assert allocator_stats()["allocations"] == before + 1
    expected = torch.arange(1 << 20, device="cuda:0", dtype=torch.int64) % 251
    assert torch.equal(tensor[: 1 << 20].to(torch.int64), expected)

    if torch.cuda.device_count() > 1:
        second_pool = kv_mem_pool(1)
        assert second_pool is kv_mem_pool("cuda:1")
        assert second_pool is not first_pool
