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
import os

import pytest


def ops_test_world_size():
    """World size for the shared ops worker pool.

    8 by default; overridable via MORI_TEST_WORLD_SIZE so the suite can run on
    a node where only a subset of the GPUs is free (pair it with
    HIP_VISIBLE_DEVICES to select which ones).
    """
    return int(os.environ.get("MORI_TEST_WORLD_SIZE", "8"))


@pytest.fixture(scope="session")
def torch_dist_process_manager():
    """Single shared worker pool for all ops tests."""
    from tests.python.ops.dispatch_combine_test_utils import (
        start_torch_dist_process_manager,
    )

    manager = start_torch_dist_process_manager(world_size=ops_test_world_size())
    yield manager
    manager.shutdown()


@pytest.fixture(scope="session", autouse=True)
def set_shmem_heap_size():
    # Set shmem heap size for all tests in this directory.
    # 32G is the minimum required by the large-token internode_v1 test
    # (~17.7 GB per rank).  Using a session-scoped fixture (instead of
    # pytest_configure) so the override is scoped to ops/ tests only and
    # does not bleed into other test directories (e.g. shmem/) when the
    # full suite is collected together.
    #
    # This fixture used to overwrite MORI_SHMEM_HEAP_SIZE unconditionally,
    # which silently clobbered a smaller value set by the caller -- so on a
    # contended node, a run explicitly asking for a heap that FITS still tried
    # 32G/rank and died in shmem init with "hip failed with out of memory",
    # looking like a failure of the code under test. An explicit caller value
    # now wins; the 32G default only applies when nothing was set.
    prev = os.environ.get("MORI_SHMEM_HEAP_SIZE")
    if prev is None:
        os.environ["MORI_SHMEM_HEAP_SIZE"] = "32G"
    yield
    if prev is None:
        os.environ.pop("MORI_SHMEM_HEAP_SIZE", None)
    else:
        os.environ["MORI_SHMEM_HEAP_SIZE"] = prev
