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

from mori.cpp import UMBPMasterClient, UMBPTierType, UMBPExternalKvNodeMatch


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


@contextlib.contextmanager
def _registered_client(master_address: str, node_id: str, caps=None):
    """Context manager that yields a UMBPMasterClient registered with the master."""
    client = UMBPMasterClient(master_address, node_id=node_id, node_address=node_id)
    client.register_self(caps or _DEFAULT_CAPS)
    try:
        yield client
    finally:
        with contextlib.suppress(Exception):
            client.unregister_self()


@pytest.fixture(scope="module")
def master_address():
    if not MASTER_BIN.is_file():
        pytest.skip(f"UMBP master binary not found: {MASTER_BIN}")

    port = _get_free_port()
    metrics_port = _get_free_port()
    address = f"localhost:{port}"

    proc = subprocess.Popen(
        [str(MASTER_BIN), address, str(metrics_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not _wait_for_port("localhost", port, timeout=10.0):
        proc.terminate()
        proc.wait()
        pytest.fail(f"UMBP master server did not start within 10s on port {port}")

    yield address

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


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
    with pytest.raises(RuntimeError):
        client.unregister_self()


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


def test_report_empty_hashes_raises(master_address):
    client = UMBPMasterClient(master_address)
    with pytest.raises(RuntimeError, match="empty"):
        client.report_external_kv_blocks("any-node", [], UMBPTierType.DRAM)


def test_revoke_empty_hashes_raises(master_address):
    client = UMBPMasterClient(master_address)
    with pytest.raises(RuntimeError, match="empty"):
        client.revoke_external_kv_blocks("any-node", [])


def test_report_and_match_external_kv_dram(master_address):
    node_id = "int-test-node-dram"
    hashes = _make_hashes("dram", 8)

    with _registered_client(master_address, node_id) as client:
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.DRAM)

        matches = client.match_external_kv(hashes)
        assert len(matches) == 1
        match = matches[0]
        assert match.node_id == node_id
        assert match.tier == UMBPTierType.DRAM
        assert set(match.matched_hashes) == set(hashes)


def test_report_and_match_external_kv_hbm(master_address):
    node_id = "int-test-node-hbm"
    hashes = _make_hashes("hbm", 4)
    caps = {UMBPTierType.HBM: (_1GB, _1GB)}

    with _registered_client(master_address, node_id, caps) as client:
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.HBM)

        matches = client.match_external_kv(hashes)
        node_ids = {m.node_id for m in matches}
        assert node_id in node_ids
        for m in matches:
            if m.node_id == node_id:
                assert m.tier == UMBPTierType.HBM
                assert set(m.matched_hashes).issubset(set(hashes))


def test_match_returns_only_subset_of_queried_hashes(master_address):
    node_id = "int-test-node-subset"
    reported = _make_hashes("subset-reported", 6)
    extra = _make_hashes("subset-extra", 4)

    with _registered_client(master_address, node_id) as client:
        client.report_external_kv_blocks(node_id, reported, UMBPTierType.DRAM)

        query = reported[:3] + extra
        matches = client.match_external_kv(query)

        all_matched = {h for m in matches for h in m.matched_hashes}
        assert all_matched.issubset(set(reported))
        assert not all_matched.intersection(set(extra))


def test_revoke_removes_hashes_from_index(master_address):
    node_id = "int-test-node-revoke"
    hashes = _make_hashes("revoke", 6)

    with _registered_client(master_address, node_id) as client:
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.DRAM)
        assert any(m.node_id == node_id for m in client.match_external_kv(hashes))

        client.revoke_external_kv_blocks(node_id, hashes)

        node_ids_after = {m.node_id for m in client.match_external_kv(hashes)}
        assert node_id not in node_ids_after


def test_revoke_partial_hashes(master_address):
    node_id = "int-test-node-partial-revoke"
    hashes = _make_hashes("partial", 10)
    to_revoke = hashes[:4]
    to_keep = hashes[4:]

    with _registered_client(master_address, node_id) as client:
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.DRAM)
        client.revoke_external_kv_blocks(node_id, to_revoke)

        kept_matched = {
            h
            for m in client.match_external_kv(to_keep)
            if m.node_id == node_id
            for h in m.matched_hashes
        }
        assert kept_matched == set(to_keep)

        revoked_matched = {
            h
            for m in client.match_external_kv(to_revoke)
            if m.node_id == node_id
            for h in m.matched_hashes
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
        ca.report_external_kv_blocks(node_a, hashes, UMBPTierType.DRAM)
        cb.report_external_kv_blocks(node_b, hashes, UMBPTierType.HBM)

        matches = ca.match_external_kv(hashes)
        matched_nodes = {m.node_id for m in matches}
        assert node_a in matched_nodes
        assert node_b in matched_nodes


def test_report_overwrites_tier_for_same_node(master_address):
    node_id = "int-test-node-overwrite"
    hashes = _make_hashes("overwrite", 3)

    with _registered_client(master_address, node_id) as client:
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.DRAM)
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.HBM)

        node_matches = [
            m for m in client.match_external_kv(hashes) if m.node_id == node_id
        ]
        assert len(node_matches) >= 1


def test_revoke_nonexistent_hashes_does_not_raise(master_address):
    # Revoking hashes that were never reported is a no-op.
    hashes = _make_hashes("nonexistent", 3)
    client = UMBPMasterClient(master_address)
    client.revoke_external_kv_blocks("any-node", hashes)


def test_external_kv_node_match_repr():
    match = UMBPExternalKvNodeMatch()
    match.node_id = "my-node"
    match.matched_hashes = ["h1", "h2"]
    match.tier = UMBPTierType.HBM
    r = repr(match)
    assert "my-node" in r


def test_match_external_kv_connection_refused_raises():
    client = UMBPMasterClient("localhost:19998")
    with pytest.raises(RuntimeError):
        client.match_external_kv(["some-hash"])


def test_report_external_kv_connection_refused_raises():
    client = UMBPMasterClient("localhost:19997")
    with pytest.raises(RuntimeError):
        client.report_external_kv_blocks("node", ["hash"], UMBPTierType.DRAM)


def test_revoke_external_kv_connection_refused_raises():
    client = UMBPMasterClient("localhost:19996")
    with pytest.raises(RuntimeError):
        client.revoke_external_kv_blocks("node", ["hash"])


def test_large_batch_report_and_match(master_address):
    node_id = "int-test-node-large"
    hashes = _make_hashes("large", 500)

    with _registered_client(master_address, node_id) as client:
        client.report_external_kv_blocks(node_id, hashes, UMBPTierType.DRAM)

        matches = client.match_external_kv(hashes)
        all_matched = {h for m in matches for h in m.matched_hashes}
        assert set(hashes).issubset(all_matched)
