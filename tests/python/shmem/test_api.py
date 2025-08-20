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
import mori
from tests.python.utils import TorchDistContext, get_free_port
import torch


def _test_torch_init(rank, world_size, port):
    with TorchDistContext(rank=rank, world_size=world_size, master_port=port) as ctx:
        mori.shmem.shmem_torch_process_group_init("default")
        assert rank == mori.shmem.shmem_mype()
        assert world_size == mori.shmem.shmem_npes()
        mori.shmem.shmem_finalize()


@pytest.mark.parametrize("world_size", (8,))
def test_torch_init(world_size):
    torch.multiprocessing.spawn(
        _test_torch_init,
        args=(world_size, get_free_port()),
        nprocs=world_size,
        join=True,
    )
