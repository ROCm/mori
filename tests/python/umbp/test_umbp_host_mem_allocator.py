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
import ctypes
import os

import pytest

umbp = pytest.importorskip("mori.umbp")

UMBPHostBufferBacking = getattr(umbp, "UMBPHostBufferBacking", None)
UMBPHostMemAllocator = getattr(umbp, "UMBPHostMemAllocator", None)
if UMBPHostBufferBacking is None or UMBPHostMemAllocator is None:
    pytest.skip(
        "mori.umbp was built without UMBP host memory bindings",
        allow_module_level=True,
    )


def _page_size() -> int:
    return os.sysconf("SC_PAGE_SIZE")


def _touch(handle, value: int = 0x5A) -> None:
    assert handle
    data = (ctypes.c_ubyte * handle.requested_size).from_address(handle.ptr)
    data[0] = value
    data[handle.requested_size - 1] = value
    assert data[0] == value
    assert data[handle.requested_size - 1] == value


def test_anonymous_alloc_free_round_trip():
    allocator = UMBPHostMemAllocator()

    handle = allocator.alloc(
        12345,
        UMBPHostBufferBacking.Anonymous,
        hugepage_size=2 * 1024 * 1024,
        numa_node=-1,
        prefault=True,
    )

    assert handle
    assert handle.ptr != 0
    assert handle.requested_size == 12345
    assert handle.actual_backing == UMBPHostBufferBacking.Anonymous
    assert handle.actual_alignment == _page_size()
    assert handle.mapped_size >= handle.requested_size
    assert handle.mapped_size % _page_size() == 0
    assert handle.ptr % handle.actual_alignment == 0
    _touch(handle)

    allocator.free(handle)

    assert not handle
    assert handle.ptr == 0
    assert handle.requested_size == 0
    assert handle.mapped_size == 0


def test_hugetlb_request_is_writable_even_when_demoted():
    allocator = UMBPHostMemAllocator()

    handle = allocator.alloc(
        64 * 1024,
        UMBPHostBufferBacking.AnonymousHugetlb,
        hugepage_size=2 * 1024 * 1024,
        numa_node=-1,
        prefault=True,
    )

    assert handle
    assert handle.actual_backing in (
        UMBPHostBufferBacking.Anonymous,
        UMBPHostBufferBacking.AnonymousHugetlb,
    )
    if handle.actual_backing == UMBPHostBufferBacking.AnonymousHugetlb:
        assert handle.actual_alignment == 2 * 1024 * 1024
    else:
        assert handle.actual_alignment == _page_size()
    assert handle.ptr % handle.actual_alignment == 0
    _touch(handle, value=0xA5)

    allocator.free(handle)
    assert not handle


def test_free_is_idempotent_for_invalidated_handle():
    allocator = UMBPHostMemAllocator()
    handle = allocator.alloc(4096)

    assert handle
    allocator.free(handle)
    assert not handle
    assert handle.requested_size == 0
    assert handle.mapped_size == 0

    allocator.free(handle)
    assert not handle
    assert handle.requested_size == 0
    assert handle.mapped_size == 0
