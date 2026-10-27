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
import gc
import weakref

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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("method", ["scalar", "batch", "depth", "ranges"])
def test_configured_kv_metadata_preserves_write_api(enabled, method):
    config = UMBPConfig()
    config.dram.capacity_bytes = 1 << 20
    config.ssd.enabled = False
    config.page_size = 4096
    metadata = umbp.KvBlockMetadata(42, 42, 16, {1: ["key"]})
    calls = []

    def provider(keys):
        calls.append(keys)
        return [metadata]

    if enabled:
        # An unavailable Indexer must not fail a successful storage write.
        config.kv_indexer = umbp.KvIndexerConfig(
            "127.0.0.1:1", "worker", "http://worker:30000", provider, timeout_ms=10
        )
    client = UMBPClient(config)
    assert not hasattr(client, "register_kv_block")
    data = ctypes.create_string_buffer(b"value")
    ptr, size = ctypes.addressof(data), len(data)

    def put():
        if method == "scalar":
            return client.put_from_ptr("key", ptr, size)
        if method == "ranges":
            return all(
                client.batch_put_ranges_from_ptr(
                    ["key"], [size], [[ptr]], [[size]], [[0]]
                )
            )
        args = (["key"], [ptr], [size])
        if method == "depth":
            return all(client.batch_put_from_ptr_with_depth(*args, [0]))
        return all(client.batch_put_from_ptr(*args))

    if enabled:
        with pytest.raises(ValueError, match="cycle"):
            put()
        assert not client.exists("key")
    metadata = umbp.KvBlockMetadata(42, None, 16, {1: ["key"]})
    assert put()
    assert calls == ([["key"], ["key"]] if enabled else [])
    out = ctypes.create_string_buffer(size)
    assert client.get_into_ptr("key", ctypes.addressof(out), size)
    assert out.raw == data.raw
    assert client.clear()


def test_indexer_requires_callable_metadata_provider():
    with pytest.raises(TypeError):
        umbp.KvIndexerConfig("127.0.0.1:1", "worker", "http://worker:30000", None)


def test_bound_metadata_provider_does_not_retain_client_owner():
    class Adapter:
        def __init__(self):
            config = UMBPConfig()
            config.dram.capacity_bytes = 1 << 20
            config.ssd.enabled = False
            config.page_size = 4096
            config.kv_indexer = umbp.KvIndexerConfig(
                "127.0.0.1:1",
                "worker",
                "http://worker:30000",
                self.metadata,
                timeout_ms=10,
            )
            self.client = UMBPClient(config)

        def metadata(self, keys):
            return [umbp.KvBlockMetadata(42, None, 16, {1: keys})]

    adapter = Adapter()
    client = adapter.client
    data = ctypes.create_string_buffer(b"value")
    assert client.put_from_ptr("key", ctypes.addressof(data), len(data))
    ref = weakref.ref(adapter)
    del adapter
    gc.collect()
    assert ref() is None
    with pytest.raises(RuntimeError, match="owner was released"):
        client.put_from_ptr("key", ctypes.addressof(data), len(data))


def test_batch_put_get_ranges_round_trip_and_shape_failure():
    client = _make_client()
    a = (ctypes.c_ubyte * 4)(*b"AAAA")
    b = (ctypes.c_ubyte * 4)(*b"BBBB")
    c = (ctypes.c_ubyte * 4)(*b"CCCC")

    assert client.batch_put_ranges_from_ptr(
        ["range_key"],
        [12],
        [[ctypes.addressof(b), ctypes.addressof(c), ctypes.addressof(a)]],
        [[4, 4, 4]],
        [[4, 8, 0]],
    ) == [True]
    assert client.exists("range_key")

    prefix = (ctypes.c_ubyte * 4)()
    suffix = (ctypes.c_ubyte * 4)()
    assert client.batch_get_ranges_into_ptr(
        ["range_key"],
        [[ctypes.addressof(suffix), ctypes.addressof(prefix)]],
        [[4, 4]],
        [[8, 0]],
    ) == [True]
    assert bytes(prefix) == b"AAAA"
    assert bytes(suffix) == b"CCCC"

    # A ragged inner vector is a whole-call shape error.
    assert client.batch_get_ranges_into_ptr(
        ["range_key", "missing"],
        [[ctypes.addressof(prefix)], [ctypes.addressof(suffix)]],
        [[4, 4], [4]],
        [[0], [0]],
    ) == [False, False]
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


@pytest.mark.parametrize("ssd_enabled", [False, True])
def test_no_master_clear_allows_new_ranged_put_and_restore(tmp_path, ssd_enabled):
    """A successful local clear must reopen every backend for fresh writes."""
    client = _make_client(ssd_dir=tmp_path / "reset-ssd" if ssd_enabled else None)
    key = "reset-range"
    for cycle in range(3):
        data = (ctypes.c_ubyte * 256)(*([cycle + 1] * 256))
        out = (ctypes.c_ubyte * 256)()
        assert client.batch_put_ranges_from_ptr(
            [key], [256], [[ctypes.addressof(data)]], [[256]], [[0]]
        ) == [True]
        assert client.batch_get_ranges_into_ptr(
            [key], [[ctypes.addressof(out)]], [[256]], [[0]]
        ) == [True]
        assert bytes(out) == bytes(data)
        assert client.clear()
        assert not client.exists(key)
