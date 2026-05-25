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
import contextlib
import os
import socket
import subprocess
import time
from pathlib import Path

import pytest

from mori.cpp import (
    UMBPClient,
    UMBPConfig,
    UMBPDistributedConfig,
    UMBPMasterClient,
    UMBPTierType,
    UMBPExternalKvNodeMatch,
    UMBPExternalKvHitCountEntry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MASTER_BIN = REPO_ROOT / "build/lib.linux-x86_64-cpython-310/mori/umbp_master"
MASTER_BIN = Path(os.environ.get("UMBP_MASTER_BIN", str(_DEFAULT_MASTER_BIN)))

_1GB = 1 * 1024 * 1024 * 1024
_DEFAULT_CAPS = {UMBPTierType.DRAM: (_1GB, _1GB)}


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def _make_hashes(prefix: str, count: int) -> list[str]:
    return [f"{prefix}-hash-{i:04d}" for i in range(count)]


def _start_master_process(address: str, metrics_port: int = 0) -> subprocess.Popen:
    if not MASTER_BIN.is_file():
        pytest.skip(f"UMBP master binary not found: {MASTER_BIN}")

    proc = subprocess.Popen(
        [str(MASTER_BIN), address, str(metrics_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    host, port_text = address.rsplit(":", 1)
    if not _wait_for_port(host, int(port_text), timeout=10.0):
        _stop_master_process(proc)
        pytest.fail(f"UMBP master server did not start within 10s on {address}")
    return proc


def _stop_master_process(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@contextlib.contextmanager
def _running_master(address: str, metrics_port: int = 0):
    proc = _start_master_process(address, metrics_port)
    try:
        yield proc
    finally:
        _stop_master_process(proc)


def _hit_counts(client: UMBPMasterClient, hashes: list[str]) -> dict[str, int]:
    return {
        entry.hash: entry.hit_count_total
        for entry in client.get_external_kv_hit_counts(hashes)
    }


@contextlib.contextmanager
def _registered_client(master_address: str, node_id: str, caps=None):
    """Context manager that yields a distributed UMBPClient registered with the master."""
    cfg = UMBPConfig()
    cfg.dram.capacity_bytes = 8 * 1024 * 1024
    cfg.ssd.enabled = False
    dist = UMBPDistributedConfig()
    dist.master_config.master_address = master_address
    dist.master_config.node_id = node_id
    dist.master_config.node_address = node_id
    dist.master_config.auto_heartbeat = True
    cfg.distributed = dist
    client = UMBPClient(cfg)
    try:
        yield client
    finally:
        with contextlib.suppress(Exception):
            client.close()


@contextlib.contextmanager
def _registered_master_client(master_address: str, node_id: str, caps=None):
    client = UMBPMasterClient(master_address, node_id=node_id, node_address=node_id)
    client.register_self(caps or _DEFAULT_CAPS)
    try:
        yield client
    finally:
        with contextlib.suppress(Exception):
            client.unregister_self()


def _bind(client, hashes: list[str], tier) -> None:
    assert client.bind_external_hashes(hashes, tier)
    assert client.flush_external_queue()


def _unbind(client, hashes: list[str], tier) -> None:
    assert client.unbind_external_hashes(hashes, tier)
    assert client.flush_external_queue()


def _clear_tier(client, tier) -> None:
    assert client.unbind_all_external_hashes_at_tier(tier)
    assert client.flush_external_queue()


def _flush_external_queue_until_ok(client, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.flush_external_queue():
            return True
        time.sleep(0.2)
    return False


@pytest.fixture(scope="module")
def master_address():
    port = _get_free_port()
    metrics_port = _get_free_port()
    address = f"localhost:{port}"

    with _running_master(address, metrics_port):
        yield address


# ---------------------------------------------------------------------------
# Construction tests (no server required)
# ---------------------------------------------------------------------------


def test_master_client_construction_does_not_raise():
    # UMBPMasterClient creates a gRPC channel lazily; construction never
    # fails even if the address is unreachable.
    client = UMBPMasterClient("localhost:19999")
    assert client is not None


def test_master_client_construction_with_node_id():
    client = UMBPMasterClient("localhost:19999", node_id="n1", node_address="n1:8080")
    assert client is not None
    assert not client.is_registered()


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_register_self_and_is_registered(master_address):
    client = UMBPMasterClient(
        master_address, node_id="reg-test-node", node_address="reg-test-node:8080"
    )
    assert not client.is_registered()
    client.register_self(_DEFAULT_CAPS)
    assert client.is_registered()
    client.unregister_self()
    assert not client.is_registered()


def test_register_self_connection_refused_raises():
    client = UMBPMasterClient("localhost:19995", node_id="n", node_address="n:1")
    with pytest.raises(RuntimeError):
        client.register_self(_DEFAULT_CAPS)


def test_unregister_not_registered_raises(master_address):
    client = UMBPMasterClient(
        master_address, node_id="unreg-test-node", node_address="unreg-test-node:8080"
    )
    client.unregister_self()
    assert not client.is_registered()


# ---------------------------------------------------------------------------
# Integration tests (require live master server)
# ---------------------------------------------------------------------------


def test_match_external_kv_empty_hashes(master_address):
    client = UMBPMasterClient(master_address)
    matches = client.match_external_kv([])
    assert matches == []


def test_match_external_kv_unknown_hashes_returns_empty(master_address):
    client = UMBPMasterClient(master_address)
    hashes = _make_hashes("unknown", 5)
    matches = client.match_external_kv(hashes)
    assert matches == []


def test_external_kv_hit_counts_are_sparse_for_unknown_hashes(master_address):
    client = UMBPMasterClient(master_address)
    assert client.get_external_kv_hit_counts(_make_hashes("unknown-hit-count", 3)) == []


def test_master_client_report_external_kv_blocks_round_trip(master_address):
    node_id = "rpc-report-node"
    hashes = _make_hashes("rpc-report", 3)

    with _registered_master_client(master_address, node_id):
        client = UMBPMasterClient(master_address)
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.DRAM)

        matches = [m for m in client.match_external_kv(hashes) if m.node_id == node_id]
        assert len(matches) == 1
        assert set(matches[0].hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)


def test_master_client_revoke_external_kv_blocks_single_tier(master_address):
    node_id = "rpc-revoke-tier-node"
    hashes = _make_hashes("rpc-revoke-tier", 4)
    caps = {UMBPTierType.HBM: (_1GB, _1GB), UMBPTierType.DRAM: (_1GB, _1GB)}

    with _registered_master_client(master_address, node_id, caps):
        client = UMBPMasterClient(master_address)
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.HBM)
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.DRAM)
        client.revoke_external_kv_blocks(node_id, hashes, UMBPTierType.HBM)

        matches = [m for m in client.match_external_kv(hashes) if m.node_id == node_id]
        assert len(matches) == 1
        assert UMBPTierType.HBM not in matches[0].hashes_by_tier
        assert set(matches[0].hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)


def test_master_client_revoke_all_external_kv_blocks_at_tier(master_address):
    node_id = "rpc-revoke-all-node"
    hashes = _make_hashes("rpc-revoke-all", 8)

    with _registered_master_client(master_address, node_id):
        client = UMBPMasterClient(master_address)
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.DRAM)
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.SSD)
        client.revoke_all_external_kv_blocks_at_tier(node_id, UMBPTierType.SSD)

        matches = [m for m in client.match_external_kv(hashes) if m.node_id == node_id]
        assert len(matches) == 1
        assert UMBPTierType.SSD not in matches[0].hashes_by_tier
        assert set(matches[0].hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)


def test_master_client_report_for_unregistered_node_is_ignored(master_address):
    client = UMBPMasterClient(master_address)
    hashes = _make_hashes("rpc-ghost", 1)

    client.report_external_kv_blocks("rpc-ghost-node", hashes, UMBPTierType.DRAM)

    assert all(m.node_id != "rpc-ghost-node" for m in client.match_external_kv(hashes))


def test_master_client_report_after_unregister_is_ignored(master_address):
    node_id = "rpc-dead-node"
    hashes = _make_hashes("rpc-dead", 1)

    client = UMBPMasterClient(master_address, node_id=node_id, node_address=node_id)
    client.register_self(_DEFAULT_CAPS)
    client.unregister_self()

    client.report_external_kv_blocks(node_id, hashes, UMBPTierType.DRAM)

    assert all(m.node_id != node_id for m in client.match_external_kv(hashes))


def test_master_client_revoke_for_unknown_node_is_noop(master_address):
    client = UMBPMasterClient(master_address)
    client.revoke_external_kv_blocks(
        "rpc-unknown-revoke-node",
        _make_hashes("rpc-unknown-revoke", 2),
        UMBPTierType.DRAM,
    )
    client.revoke_all_external_kv_blocks_at_tier(
        "rpc-unknown-revoke-node", UMBPTierType.DRAM
    )


def test_master_client_report_empty_node_id_raises(master_address):
    client = UMBPMasterClient(master_address)
    with pytest.raises(RuntimeError, match="node_id"):
        client.report_external_kv_blocks("", ["rpc-empty-node-hash"], UMBPTierType.DRAM)


def test_master_client_report_empty_hashes_raises(master_address):
    node_id = "rpc-empty-hashes-node"
    with _registered_master_client(master_address, node_id):
        client = UMBPMasterClient(master_address)
        with pytest.raises(RuntimeError, match="hashes"):
            client.report_external_kv_blocks(node_id, [], UMBPTierType.DRAM)


def test_umbpclient_report_external_kv_blocks_alias_is_visible(master_address):
    node_id = "alias-report-node"
    hashes = _make_hashes("alias-report", 2)

    with _registered_client(master_address, node_id) as client:
        assert client.report_external_kv_blocks(hashes, UMBPTierType.DRAM)

        query = UMBPMasterClient(master_address)
        matches = [m for m in query.match_external_kv(hashes) if m.node_id == node_id]
        assert len(matches) == 1
        assert set(matches[0].hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)


def test_umbpclient_two_arg_alias_survives_external_full_sync():
    port = _get_free_port()
    address = f"localhost:{port}"
    node_id = "alias-full-sync-node"
    hashes = _make_hashes("alias-full-sync", 2)

    proc = _start_master_process(address)
    try:
        with _registered_client(address, node_id) as client:
            assert client.report_external_kv_blocks(hashes, UMBPTierType.DRAM)

            query = UMBPMasterClient(address)
            matches = [
                m for m in query.match_external_kv(hashes) if m.node_id == node_id
            ]
            assert len(matches) == 1
            assert set(matches[0].hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)

            _stop_master_process(proc)
            proc = None

            proc = _start_master_process(address)
            assert _flush_external_queue_until_ok(client)
            assert _flush_external_queue_until_ok(client)

            query = UMBPMasterClient(address)
            matches = [
                m for m in query.match_external_kv(hashes) if m.node_id == node_id
            ]
            assert len(matches) == 1
            assert set(matches[0].hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)
    finally:
        if proc is not None:
            _stop_master_process(proc)


def test_umbpclient_revoke_external_kv_blocks_alias(master_address):
    node_id = "alias-revoke-node"
    hashes = _make_hashes("alias-revoke", 2)

    with _registered_client(master_address, node_id) as client:
        assert client.report_external_kv_blocks(hashes, UMBPTierType.DRAM)
        assert client.revoke_external_kv_blocks(hashes, UMBPTierType.DRAM)

        query = UMBPMasterClient(master_address)
        assert all(m.node_id != node_id for m in query.match_external_kv(hashes))


def test_umbpclient_revoke_all_external_kv_blocks_at_tier_alias(master_address):
    node_id = "alias-revoke-all-node"
    hashes = _make_hashes("alias-revoke-all", 3)

    with _registered_client(master_address, node_id) as client:
        assert client.report_external_kv_blocks(hashes, UMBPTierType.DRAM)
        assert client.report_external_kv_blocks(hashes, UMBPTierType.SSD)
        assert client.revoke_all_external_kv_blocks_at_tier(UMBPTierType.SSD)

        query = UMBPMasterClient(master_address)
        matches = [m for m in query.match_external_kv(hashes) if m.node_id == node_id]
        assert len(matches) == 1
        assert UMBPTierType.SSD not in matches[0].hashes_by_tier
        assert set(matches[0].hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)


def test_umbpclient_clear_removes_external_kv_immediately(master_address):
    node_id = "clear-external-node"
    hashes = _make_hashes("clear-external", 3)

    with _registered_client(master_address, node_id) as client:
        assert client.report_external_kv_blocks(hashes, UMBPTierType.DRAM)

        query = UMBPMasterClient(master_address)
        assert any(m.node_id == node_id for m in query.match_external_kv(hashes))

        assert client.clear()
        assert all(m.node_id != node_id for m in query.match_external_kv(hashes))


def test_umbpclient_clear_allows_rebind_and_second_clear(master_address):
    node_id = "clear-rebind-node"
    hashes = _make_hashes("clear-rebind", 2)

    with _registered_client(master_address, node_id) as client:
        assert client.report_external_kv_blocks(hashes, UMBPTierType.DRAM)
        assert client.clear()

        query = UMBPMasterClient(master_address)
        assert all(m.node_id != node_id for m in query.match_external_kv(hashes))

        assert client.report_external_kv_blocks(hashes, UMBPTierType.DRAM)
        assert any(m.node_id == node_id for m in query.match_external_kv(hashes))

        assert client.clear()
        assert all(m.node_id != node_id for m in query.match_external_kv(hashes))


def test_bind_empty_hashes_is_noop(master_address):
    with _registered_client(master_address, "empty-bind-node") as client:
        _bind(client, [], UMBPTierType.DRAM)


def test_unbind_empty_hashes_is_noop(master_address):
    with _registered_client(master_address, "empty-unbind-node") as client:
        _unbind(client, [], UMBPTierType.DRAM)


def test_report_and_match_external_kv_dram(master_address):
    node_id = "int-test-node-dram"
    hashes = _make_hashes("dram", 8)

    with _registered_client(master_address, node_id) as client:
        _bind(client, hashes, UMBPTierType.DRAM)

        matches = client.match_external_kv(hashes)
        assert len(matches) == 1
        match = matches[0]
        assert match.node_id == node_id
        assert match.matched_hash_count() == len(hashes)
        assert UMBPTierType.DRAM in match.hashes_by_tier
        assert set(match.hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)


def test_match_external_kv_count_as_hit_false_does_not_record(master_address):
    node_id = "hit-count-disabled-node"
    hashes = _make_hashes("hit-count-disabled", 2)

    with _registered_client(master_address, node_id) as client:
        _bind(client, hashes, UMBPTierType.DRAM)

        for _ in range(3):
            assert client.match_external_kv(hashes)

        assert client.get_external_kv_hit_counts(hashes) == []


def test_match_external_kv_count_as_hit_records_only_matched_unique_hashes(
    master_address,
):
    node_a = "hit-count-node-a"
    node_b = "hit-count-node-b"
    hot_hash = "hit-count-unique-hot"
    missing_hash = "hit-count-unique-missing"

    with (
        _registered_client(master_address, node_a) as ca,
        _registered_client(master_address, node_b) as cb,
    ):
        _bind(ca, [hot_hash], UMBPTierType.DRAM)
        _bind(ca, [hot_hash], UMBPTierType.HBM)
        _bind(cb, [hot_hash], UMBPTierType.DRAM)

        matches = ca.match_external_kv(
            [hot_hash, hot_hash, hot_hash, missing_hash], count_as_hit=True
        )
        assert {m.node_id for m in matches} == {node_a, node_b}

        counts = _hit_counts(ca, [hot_hash, hot_hash, missing_hash])
        assert counts == {hot_hash: 1}


def test_external_kv_hit_counts_accumulate_and_survive_revoke(master_address):
    node_id = "hit-count-revoke-node"
    hot_hash = "hit-count-revoke-hot"

    with _registered_client(master_address, node_id) as client:
        _bind(client, [hot_hash], UMBPTierType.DRAM)
        for _ in range(4):
            client.match_external_kv([hot_hash], count_as_hit=True)

        assert _hit_counts(client, [hot_hash]) == {hot_hash: 4}

        _unbind(client, [hot_hash], UMBPTierType.DRAM)
        assert client.match_external_kv([hot_hash], count_as_hit=True) == []
        assert _hit_counts(client, [hot_hash]) == {hot_hash: 4}


def test_get_external_kv_hit_counts_rejects_oversized_batch(master_address):
    client = UMBPMasterClient(master_address)
    max_batch = int(os.environ.get("UMBP_HIT_QUERY_MAX_BATCH", "4096"))
    with pytest.raises(RuntimeError, match="UMBP_HIT_QUERY_MAX_BATCH"):
        client.get_external_kv_hit_counts(
            _make_hashes("too-many-hit-counts", max_batch + 1)
        )


def test_report_and_match_external_kv_hbm(master_address):
    node_id = "int-test-node-hbm"
    hashes = _make_hashes("hbm", 4)
    caps = {UMBPTierType.HBM: (_1GB, _1GB)}

    with _registered_client(master_address, node_id, caps) as client:
        _bind(client, hashes, UMBPTierType.HBM)

        matches = client.match_external_kv(hashes)
        node_ids = {m.node_id for m in matches}
        assert node_id in node_ids
        for m in matches:
            if m.node_id == node_id:
                assert UMBPTierType.HBM in m.hashes_by_tier
                assert set(m.hashes_by_tier[UMBPTierType.HBM]).issubset(set(hashes))


def test_match_returns_only_subset_of_queried_hashes(master_address):
    node_id = "int-test-node-subset"
    reported = _make_hashes("subset-reported", 6)
    extra = _make_hashes("subset-extra", 4)

    with _registered_client(master_address, node_id) as client:
        _bind(client, reported, UMBPTierType.DRAM)

        query = reported[:3] + extra
        matches = client.match_external_kv(query)

        all_matched = {
            h for m in matches for hs in m.hashes_by_tier.values() for h in hs
        }
        assert all_matched.issubset(set(reported))
        assert not all_matched.intersection(set(extra))


def test_revoke_removes_hashes_from_index(master_address):
    node_id = "int-test-node-revoke"
    hashes = _make_hashes("revoke", 6)

    with _registered_client(master_address, node_id) as client:
        _bind(client, hashes, UMBPTierType.DRAM)
        assert any(m.node_id == node_id for m in client.match_external_kv(hashes))

        _unbind(client, hashes, UMBPTierType.DRAM)

        node_ids_after = {m.node_id for m in client.match_external_kv(hashes)}
        assert node_id not in node_ids_after


def test_revoke_partial_hashes(master_address):
    node_id = "int-test-node-partial-revoke"
    hashes = _make_hashes("partial", 10)
    to_revoke = hashes[:4]
    to_keep = hashes[4:]

    with _registered_client(master_address, node_id) as client:
        _bind(client, hashes, UMBPTierType.DRAM)
        _unbind(client, to_revoke, UMBPTierType.DRAM)

        kept_matched = {
            h
            for m in client.match_external_kv(to_keep)
            if m.node_id == node_id
            for hs in m.hashes_by_tier.values()
            for h in hs
        }
        assert kept_matched == set(to_keep)

        revoked_matched = {
            h
            for m in client.match_external_kv(to_revoke)
            if m.node_id == node_id
            for hs in m.hashes_by_tier.values()
            for h in hs
        }
        assert revoked_matched == set()


def test_multiple_nodes_report_same_hashes(master_address):
    hashes = _make_hashes("shared", 5)
    node_a = "int-test-node-multi-a"
    node_b = "int-test-node-multi-b"

    with (
        _registered_client(master_address, node_a) as ca,
        _registered_client(master_address, node_b) as cb,
    ):
        _bind(ca, hashes, UMBPTierType.DRAM)
        _bind(cb, hashes, UMBPTierType.HBM)

        matches = ca.match_external_kv(hashes)
        matched_nodes = {m.node_id for m in matches}
        assert node_a in matched_nodes
        assert node_b in matched_nodes


def test_report_overwrites_tier_for_same_node(master_address):
    node_id = "int-test-node-overwrite"
    hashes = _make_hashes("overwrite", 3)

    with _registered_client(master_address, node_id) as client:
        _bind(client, hashes, UMBPTierType.DRAM)
        _bind(client, hashes, UMBPTierType.HBM)

        node_matches = [
            m for m in client.match_external_kv(hashes) if m.node_id == node_id
        ]
        assert len(node_matches) >= 1


def test_revoke_nonexistent_hashes_does_not_raise(master_address):
    # Revoking hashes that were never reported is a no-op.
    hashes = _make_hashes("nonexistent", 3)
    with _registered_client(master_address, "nonexistent-revoke-node") as client:
        _unbind(client, hashes, UMBPTierType.DRAM)


def test_external_kv_node_match_repr():
    match = UMBPExternalKvNodeMatch()
    match.node_id = "my-node"
    match.hashes_by_tier = {UMBPTierType.HBM: ["h1", "h2"]}
    r = repr(match)
    assert "my-node" in r
    assert match.matched_hash_count() == 2


def test_external_kv_hit_count_entry_repr():
    entry = UMBPExternalKvHitCountEntry()
    entry.hash = "my-hash"
    entry.hit_count_total = 7
    r = repr(entry)
    assert "my-hash" in r
    assert "7" in r


def test_match_external_kv_connection_refused_raises():
    client = UMBPMasterClient("localhost:19998")
    with pytest.raises(RuntimeError):
        client.match_external_kv(["some-hash"])


def test_bind_external_kv_connection_refused_raises():
    cfg = UMBPConfig()
    cfg.dram.capacity_bytes = 8 * 1024 * 1024
    cfg.ssd.enabled = False
    dist = UMBPDistributedConfig()
    dist.master_config.master_address = "localhost:19997"
    dist.master_config.node_id = "bind-refused"
    dist.master_config.node_address = "bind-refused"
    cfg.distributed = dist
    with pytest.raises(RuntimeError):
        UMBPClient(cfg)


def test_large_batch_report_and_match(master_address):
    node_id = "int-test-node-large"
    hashes = _make_hashes("large", 500)

    with _registered_client(master_address, node_id) as client:
        _bind(client, hashes, UMBPTierType.DRAM)

        matches = client.match_external_kv(hashes)
        all_matched = {
            h for m in matches for hs in m.hashes_by_tier.values() for h in hs
        }
        assert set(hashes).issubset(all_matched)


def test_register_is_additive_across_tiers(master_address):
    # Re-reporting the same hash at a new tier must keep both buckets.
    node_id = "int-test-node-additive"
    caps = {UMBPTierType.HBM: (_1GB, _1GB), UMBPTierType.DRAM: (_1GB, _1GB)}
    hashes = _make_hashes("additive", 4)

    with _registered_client(master_address, node_id, caps) as client:
        _bind(client, hashes, UMBPTierType.HBM)
        _bind(client, hashes, UMBPTierType.DRAM)

        matches = [m for m in client.match_external_kv(hashes) if m.node_id == node_id]
        assert len(matches) == 1
        m = matches[0]
        assert UMBPTierType.HBM in m.hashes_by_tier
        assert UMBPTierType.DRAM in m.hashes_by_tier
        assert set(m.hashes_by_tier[UMBPTierType.HBM]) == set(hashes)
        assert set(m.hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)


def test_revoke_drops_only_one_tier(master_address):
    # Revoking from one tier must leave other tier buckets intact.
    node_id = "int-test-node-tier-revoke"
    caps = {UMBPTierType.HBM: (_1GB, _1GB), UMBPTierType.DRAM: (_1GB, _1GB)}
    hashes = _make_hashes("tier-revoke", 3)

    with _registered_client(master_address, node_id, caps) as client:
        _bind(client, hashes, UMBPTierType.HBM)
        _bind(client, hashes, UMBPTierType.DRAM)

        _unbind(client, hashes, UMBPTierType.HBM)

        matches = [m for m in client.match_external_kv(hashes) if m.node_id == node_id]
        assert len(matches) == 1
        m = matches[0]
        assert UMBPTierType.HBM not in m.hashes_by_tier
        assert UMBPTierType.DRAM in m.hashes_by_tier
        assert set(m.hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)


def test_revoke_all_at_tier_bulk(master_address):
    # Bulk-revoke every hash at a tier; other tiers untouched.
    node_id = "int-test-node-bulk-revoke"
    caps = {UMBPTierType.DRAM: (_1GB, _1GB)}
    hashes = _make_hashes("bulk-revoke", 50)

    with _registered_client(master_address, node_id, caps) as client:
        _bind(client, hashes, UMBPTierType.DRAM)
        _bind(client, hashes, UMBPTierType.SSD)
        assert any(m.node_id == node_id for m in client.match_external_kv(hashes))

        _clear_tier(client, UMBPTierType.SSD)

        matches = [m for m in client.match_external_kv(hashes) if m.node_id == node_id]
        assert len(matches) == 1
        assert UMBPTierType.SSD not in matches[0].hashes_by_tier
        assert UMBPTierType.DRAM in matches[0].hashes_by_tier
        assert set(matches[0].hashes_by_tier[UMBPTierType.DRAM]) == set(hashes)
