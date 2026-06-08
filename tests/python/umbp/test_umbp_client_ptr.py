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

import pytest

umbp = pytest.importorskip("mori.umbp")

UMBPClient = getattr(umbp, "UMBPClient", None)
UMBPConfig = getattr(umbp, "UMBPConfig", None)
if UMBPClient is None or UMBPConfig is None:
    pytest.skip(
        "mori.umbp was built without UMBPClient bindings", allow_module_level=True
    )


class MockPageFirstKVCache:
    """Small page-first K/V buffer used to exercise MORI pointer APIs."""

    def __init__(self, num_pages=4, page_size=1, element_size=1024):
        self.page_size = page_size
        self.element_size = element_size

        total_bytes = num_pages * 2 * element_size
        self._buffer = (ctypes.c_ubyte * total_bytes)()
        self._buffer_ptr = ctypes.addressof(self._buffer)

    def get_page_buffer_meta(self, indices):
        ptr_list = []
        pages = list(range(0, len(indices), self.page_size))

        for page_start in pages:
            page_idx = indices[page_start]
            k_ptr = self._buffer_ptr + page_idx * 2 * self.element_size
            v_ptr = k_ptr + self.element_size
            ptr_list.append(k_ptr)
            ptr_list.append(v_ptr)

        return ptr_list, [self.element_size] * len(ptr_list)

    def fill_page(self, page_idx, k_val, v_val):
        k_offset = page_idx * 2 * self.element_size
        v_offset = k_offset + self.element_size
        ctypes.memset(self._buffer_ptr + k_offset, k_val, self.element_size)
        ctypes.memset(self._buffer_ptr + v_offset, v_val, self.element_size)

    def read_page_k(self, page_idx):
        k_offset = page_idx * 2 * self.element_size
        return bytes(ctypes.string_at(self._buffer_ptr + k_offset, self.element_size))

    def read_page_v(self, page_idx):
        v_offset = page_idx * 2 * self.element_size + self.element_size
        return bytes(ctypes.string_at(self._buffer_ptr + v_offset, self.element_size))


def _make_client(dram_capacity_bytes=4 * 1024 * 1024, ssd_dir=None):
    config = UMBPConfig()
    config.dram.capacity_bytes = dram_capacity_bytes
    config.ssd.enabled = ssd_dir is not None
    if ssd_dir is not None:
        config.ssd.storage_dir = str(ssd_dir)
        config.ssd.capacity_bytes = 16 * 1024 * 1024
    return UMBPClient(config)


def _expanded_keys(keys):
    return [f"{key}_{part}" for key in keys for part in ("k", "v")]


def test_put_from_ptr_get_into_ptr_round_trip():
    client = _make_client()
    data = (ctypes.c_ubyte * 256)(*([ord("Z")] * 256))
    out = (ctypes.c_ubyte * 256)()

    assert client.put_from_ptr("legacy_key", ctypes.addressof(data), len(data))
    assert client.exists("legacy_key")
    assert client.get_into_ptr("legacy_key", ctypes.addressof(out), len(out))
    assert bytes(out) == bytes(data)

    assert client.clear()


def test_batch_put_get_multiple_page_components():
    client = _make_client()
    mem_pool = MockPageFirstKVCache(num_pages=4, page_size=1, element_size=256)

    for page_idx in range(4):
        mem_pool.fill_page(page_idx, ord("A") + page_idx, ord("a") + page_idx)

    keys = _expanded_keys([f"hash_{i}" for i in range(4)])
    ptrs, sizes = mem_pool.get_page_buffer_meta([0, 1, 2, 3])
    assert client.batch_put_from_ptr(keys, ptrs, sizes) == [True] * len(keys)

    for page_idx in range(4):
        mem_pool.fill_page(page_idx, 0, 0)

    assert client.batch_get_into_ptr(keys, ptrs, sizes) == [True] * len(keys)

    for page_idx in range(4):
        assert mem_pool.read_page_k(page_idx)[0] == ord("A") + page_idx
        assert mem_pool.read_page_v(page_idx)[0] == ord("a") + page_idx

    assert client.batch_exists_consecutive(keys + ["missing_k"]) == len(keys)
    assert client.clear()
    assert client.batch_exists_consecutive(keys) == 0


def test_repeated_put_keeps_original_value():
    client = _make_client()
    mem_pool = MockPageFirstKVCache(num_pages=1, page_size=1, element_size=256)

    keys = _expanded_keys(["dedup_key"])
    ptrs, sizes = mem_pool.get_page_buffer_meta([0])

    mem_pool.fill_page(0, ord("A"), ord("B"))
    assert client.batch_put_from_ptr(keys, ptrs, sizes) == [True, True]

    mem_pool.fill_page(0, ord("X"), ord("Y"))
    assert client.batch_put_from_ptr(keys, ptrs, sizes) == [True, True]

    mem_pool.fill_page(0, 0, 0)
    assert client.batch_get_into_ptr(keys, ptrs, sizes) == [True, True]
    assert mem_pool.read_page_k(0)[0] == ord("A")
    assert mem_pool.read_page_v(0)[0] == ord("B")

    assert client.clear()


def test_batch_get_with_ssd_enabled(tmp_path):
    client = _make_client(ssd_dir=tmp_path / "umbp_segmented")
    mem_pool = MockPageFirstKVCache(num_pages=2, page_size=1, element_size=256)

    mem_pool.fill_page(0, ord("M"), ord("N"))
    keys = _expanded_keys(["seg_hash_0"])
    ptrs, sizes = mem_pool.get_page_buffer_meta([0])

    assert client.batch_put_from_ptr(keys, ptrs, sizes) == [True, True]
    mem_pool.fill_page(0, 0, 0)
    assert client.batch_get_into_ptr(keys, ptrs, sizes) == [True, True]
    assert mem_pool.read_page_k(0)[0] == ord("M")
    assert mem_pool.read_page_v(0)[0] == ord("N")
    assert client.clear()
