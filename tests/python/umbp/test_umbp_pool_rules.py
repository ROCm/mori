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
"""Pool rules through the Python binding: classes, tables, state."""

import ctypes
import json

import pytest

umbp = pytest.importorskip("mori.umbp")

UMBPClient = getattr(umbp, "UMBPClient", None)
UMBPConfig = getattr(umbp, "UMBPConfig", None)
if UMBPClient is None or UMBPConfig is None:
    pytest.skip(
        "mori.umbp was built without UMBPClient bindings", allow_module_level=True
    )


def _make_client(dram_capacity_bytes=4 * 1024 * 1024):
    config = UMBPConfig()
    config.dram.capacity_bytes = dram_capacity_bytes
    config.ssd.enabled = False
    return UMBPClient(config)


def _put(client, key, data):
    return client.put_from_ptr(key, ctypes.addressof(data), len(data))


def test_a_declined_class_is_not_stored_until_its_rule_changes():
    client = _make_client()
    data = (ctypes.c_ubyte * 256)(*([7] * 256))

    client.set_pool_rules("state", json.dumps({"rules": {"9": {"admit": False}}}))
    assert client.batch_set_key_class("state", ["declined", "kept"], [9, 1]) == 2
    assert not _put(client, "declined", data)
    assert _put(client, "kept", data)
    assert _put(client, "unclassified", data)
    assert client.batch_exists(["declined", "kept", "unclassified"]) == [
        False,
        True,
        True,
    ]

    # Replace the table while the store runs.
    client.set_pool_rules("state", "{}")
    assert client.batch_set_key_class("state", ["declined"], [9]) == 1
    assert _put(client, "declined", data)
    assert client.exists("kept")

    assert client.clear()


def test_state_reports_tables_and_counters():
    client = _make_client()
    data = (ctypes.c_ubyte * 256)(*([7] * 256))
    out = (ctypes.c_ubyte * 256)()

    client.set_pool_rules(
        "kv", json.dumps({"policy": "from-test", "rules": {"3": {"weight": 4}}})
    )
    assert client.batch_set_key_class("kv", ["a", "b"], [3]) == 2  # one tag for all
    assert _put(client, "a", data)
    assert _put(client, "b", data)
    assert client.get_into_ptr("a", ctypes.addressof(out), len(out))

    state = json.loads(client.get_pool_rules_state())
    pool = state["pools"]["kv"]
    assert pool["table"]["policy"] == "from-test"
    assert pool["table"]["version"] == 1
    assert pool["table"]["rules"]["3"]["weight"] == 4
    counters = pool["classes"]["3"]
    assert counters["admitted"] == 2
    assert counters["reads"] == 1
    assert sum(counters["reuse_by_idle"]) == 1
    assert counters["live"]["DRAM"]["keys"] == 2
    assert len(state["idle_bucket_upper_s"]) == len(counters["reuse_by_idle"]) - 1

    assert client.clear()


def test_a_malformed_table_is_rejected_with_the_reason():
    client = _make_client()
    client.set_pool_rules("state", json.dumps({"rules": {"1": {"weight": 2}}}))

    with pytest.raises(ValueError, match="weight"):
        client.set_pool_rules("state", json.dumps({"default": {"weight": 0}}))
    with pytest.raises(ValueError, match="unknown field"):
        client.set_pool_rules("state", json.dumps({"rule": {}}))

    state = json.loads(client.get_pool_rules_state())
    table = state["pools"]["state"]["table"]
    assert table["version"] == 1, "a rejected table must leave the old one in force"
    assert table["rules"]["1"]["weight"] == 2

    assert client.clear()
