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
"""mori.io.HipGemAllocator: torch tensors in amdgpu GEM BOs, moved over RDMA.

Both engines run in this process on the same GPU, so the transfer loops back
through that GPU's NIC; this also works on rail-only fabrics.
"""
import os
from contextlib import contextmanager

import pytest
import torch

from mori.io import (
    BackendType,
    IOEngine,
    IOEngineConfig,
    RdmaBackendConfig,
    hip_gem_lookup,
    make_hip_gem_mem_pool,
)


def pinning_rdma_devices():
    """The RDMA devices the allocator may pin buffers through whose dma-buf MRs do pin
    memory: mlx5 attaches dma-bufs dynamically, which leaves a buffer evictable."""
    root = "/sys/class/infiniband"
    names = sorted(os.listdir(root)) if os.path.isdir(root) else []
    pin = os.environ.get("MORI_HIP_GEM_PIN_DEVICE")
    selected = os.environ.get("MORI_RDMA_DEVICES", "")
    if pin:
        names = [n for n in names if n == pin]
    elif selected.startswith("^"):
        excluded = {n.strip() for n in selected[1:].split(",")}
        names = [n for n in names if n not in excluded]
    elif selected:
        names = [n for n in (s.strip() for s in selected.split(",")) if n in names]

    def driver(name):
        return os.path.basename(os.path.realpath(f"{root}/{name}/device/driver"))

    return [n for n in names if "mlx5" not in driver(n)]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not pinning_rdma_devices(),
    reason="needs a GPU and an RDMA device whose dma-buf MRs pin memory (not mlx5)",
)

MiB = 1 << 20
DEVICE = torch.device("cuda", 0)


@contextmanager
def temporary_env(name, value):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


@pytest.fixture(scope="module")
def pool():
    return make_hip_gem_mem_pool(DEVICE)


def gem_empty(pool, nbytes, device=DEVICE):
    with torch.cuda.use_mem_pool(pool, device=device):
        return torch.empty(nbytes, dtype=torch.uint8, device=device)


def test_tensor_lives_in_a_gem_buffer(pool):
    t = gem_empty(pool, 64 * MiB)
    found = hip_gem_lookup(t.data_ptr())
    assert found is not None
    base, size, device = found
    assert base <= t.data_ptr() < base + size
    assert device == DEVICE.index
    plain = torch.empty(MiB, dtype=torch.uint8, device=DEVICE)
    assert hip_gem_lookup(plain.data_ptr()) is None


def test_kernels_and_copies_work(pool):
    with torch.cuda.use_mem_pool(pool, device=DEVICE):
        a = torch.randn(1024, 1024, device=DEVICE)
        b = torch.randn(1024, 1024, device=DEVICE)
        c = a @ b
    assert hip_gem_lookup(c.data_ptr()) is not None
    torch.testing.assert_close(c.cpu(), a.cpu() @ b.cpu(), rtol=1e-3, atol=1e-3)


def test_mem_get_info_covers_the_buffer(pool):
    torch.cuda.synchronize(DEVICE)
    free_before, _ = torch.cuda.mem_get_info(DEVICE)
    t = gem_empty(pool, 1024 * MiB)
    torch.cuda.synchronize(DEVICE)
    free_after, _ = torch.cuda.mem_get_info(DEVICE)
    # The BO is pinned when it is allocated, so KFD's free VRAM drops right away.
    assert free_before - free_after >= 1000 * MiB
    del t


@pytest.fixture(scope="module")
def engines():
    config = IOEngineConfig(host="127.0.0.1", port=0)
    initiator = IOEngine(key="hip_gem_initiator", config=config)
    target = IOEngine(key="hip_gem_target", config=config)
    rdma = RdmaBackendConfig(
        qp_per_transfer=2,
        post_batch_size=-1,
        num_worker_threads=1,
        enable_notification=True,
    )
    with temporary_env("MORI_DISABLE_AUTO_XGMI", "1"):
        initiator.create_backend(BackendType.RDMA, rdma)
        target.create_backend(BackendType.RDMA, rdma)
    initiator.register_remote_engine(target.get_engine_desc())
    target.register_remote_engine(initiator.get_engine_desc())
    yield initiator, target


def write_and_check(engines, src, dst, nbytes):
    initiator, target = engines
    src_mem = initiator.register_torch_tensor(src)
    dst_mem = target.register_torch_tensor(dst)
    status = initiator.write(
        src_mem, 0, dst_mem, 0, nbytes, initiator.allocate_transfer_uid()
    )
    while status.InProgress():
        pass
    assert status.Succeeded(), status.Message()
    torch.cuda.synchronize(DEVICE)
    assert torch.equal(src[:nbytes].cpu(), dst[:nbytes].cpu())
    initiator.deregister_memory(src_mem)
    target.deregister_memory(dst_mem)


@pytest.mark.parametrize(
    "src_kind,dst_kind", [("gem", "gem"), ("gem", "plain"), ("plain", "gem")]
)
def test_rdma_write(engines, pool, src_kind, dst_kind):
    nbytes = 48 * MiB + 4096

    def make(kind):
        if kind == "gem":
            return gem_empty(pool, nbytes)
        return torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)

    src, dst = make(src_kind), make(dst_kind)
    src.copy_(torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=DEVICE))
    dst.zero_()
    write_and_check(engines, src, dst, nbytes)


def test_rdma_write_into_a_sub_buffer(engines, pool):
    # A per-layer view of a pooled KV buffer starts inside the GEM allocation.
    big = gem_empty(pool, 128 * MiB)
    big.zero_()
    src = gem_empty(pool, 16 * MiB)
    src.copy_(torch.randint(1, 256, (16 * MiB,), dtype=torch.uint8, device=DEVICE))
    dst = big[64 * MiB : 80 * MiB]
    write_and_check(engines, src, dst, 16 * MiB)
    assert int(big[: 64 * MiB].count_nonzero()) == 0
    assert int(big[80 * MiB :].count_nonzero()) == 0


def test_buffer_on_another_gpu_reports_that_gpu():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible GPUs")
    other = torch.device("cuda", 1)
    other_pool = make_hip_gem_mem_pool(other)  # a pool serves the device it was made on
    t = gem_empty(other_pool, 16 * MiB, device=other)
    found = hip_gem_lookup(t.data_ptr())
    assert found is not None
    assert found[2] == 1
