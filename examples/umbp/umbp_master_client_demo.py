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
"""
End-to-end demo: start an umbp_master server subprocess, exercise
UMBPMasterClient against it, then shut down cleanly.

Run from the repo root:

    python examples/umbp/umbp_master_client_demo.py

Pass the binary path explicitly:

    python examples/umbp/umbp_master_client_demo.py /path/to/umbp_master

Falls back to UMBP_MASTER_BIN env var, then build/src/umbp/umbp_master.
"""

import argparse
import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the master binary
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BIN = _REPO_ROOT / "build/src/umbp/umbp_master"


def _parse_args() -> Path:
    parser = argparse.ArgumentParser(description="UMBP master + client end-to-end demo")
    parser.add_argument(
        "binary",
        nargs="?",
        default=os.environ.get("UMBP_MASTER_BIN", str(_DEFAULT_BIN)),
        help="path to the umbp_master binary (default: %(default)s)",
    )
    return Path(parser.parse_args().binary)


MASTER_BIN = _parse_args()

if not MASTER_BIN.is_file():
    sys.exit(
        f"[ERROR] umbp_master binary not found at {MASTER_BIN}\n"
        "Build it with:  mkdir -p build && cd build && cmake .. -DUMBP=ON && make -j umbp_master\n"
        "Or pass the path directly:  python examples/umbp/umbp_master_client_demo.py /path/to/umbp_master"
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
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


@contextlib.contextmanager
def master_server():
    """Start the umbp_master subprocess and yield its gRPC address."""
    grpc_port = _free_port()
    metrics_port = _free_port()
    address = f"localhost:{grpc_port}"

    print(f"[master] starting {MASTER_BIN} {address} (metrics :{metrics_port})")
    proc = subprocess.Popen(
        [str(MASTER_BIN), address, str(metrics_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    if not _wait_for_port("localhost", grpc_port, timeout=10.0):
        proc.terminate()
        out, _ = proc.communicate(timeout=5)
        sys.exit(f"[ERROR] master did not start in time.\nOutput:\n{out.decode()}")

    print(f"[master] ready on {address}")
    try:
        yield address
    finally:
        print("[master] shutting down …")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("[master] stopped")


@contextlib.contextmanager
def registered_client(master_address: str, node_id: str, tier_caps: dict):
    """Yield a UMBPMasterClient that is registered and auto-unregisters on exit."""
    from mori.cpp import UMBPMasterClient

    client = UMBPMasterClient(master_address, node_id=node_id, node_address=node_id)
    client.register_self(tier_caps)
    print(f"[client] {node_id!r} registered  (is_registered={client.is_registered()})")
    try:
        yield client
    finally:
        with contextlib.suppress(Exception):
            client.unregister_self()
        print(f"[client] {node_id!r} unregistered")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run_demo(master_address: str) -> None:
    from mori.cpp import UMBPMasterClient, UMBPTierType

    _1GB = 1 * 1024 * 1024 * 1024
    DRAM_CAPS = {UMBPTierType.DRAM: (_1GB, _1GB)}
    HBM_CAPS = {UMBPTierType.HBM: (_1GB, _1GB)}

    # ------------------------------------------------------------------
    # Scenario 1 – single node: report then match
    # ------------------------------------------------------------------
    print("\n--- Scenario 1: report and match (single node) ---")
    hashes_a = [f"block-{i:04d}" for i in range(5)]

    with registered_client(master_address, "node-a", DRAM_CAPS) as ca:
        ca.report_external_kv_blocks("node-a", hashes_a, UMBPTierType.DRAM)
        print(f"[node-a] reported {len(hashes_a)} blocks on DRAM")

        matches = ca.match_external_kv(hashes_a)
        assert len(matches) == 1, f"expected 1 match, got {len(matches)}"
        m = matches[0]
        print(
            f"[match]  node_id={m.node_id}  tier={m.tier}  matched={len(m.matched_hashes)} blocks"
        )
        assert m.node_id == "node-a"
        assert m.tier == UMBPTierType.DRAM
        assert set(m.matched_hashes) == set(hashes_a)

    # ------------------------------------------------------------------
    # Scenario 2 – two nodes, same hashes, different tiers
    # ------------------------------------------------------------------
    print("\n--- Scenario 2: multi-node, mixed tiers ---")
    shared_hashes = [f"shared-{i:04d}" for i in range(4)]

    with (
        registered_client(master_address, "node-b", DRAM_CAPS) as cb,
        registered_client(master_address, "node-c", HBM_CAPS) as cc,
    ):

        cb.report_external_kv_blocks("node-b", shared_hashes, UMBPTierType.DRAM)
        cc.report_external_kv_blocks("node-c", shared_hashes, UMBPTierType.HBM)
        print(f"[node-b] reported {len(shared_hashes)} blocks on DRAM")
        print(f"[node-c] reported {len(shared_hashes)} blocks on HBM")

        matches = cb.match_external_kv(shared_hashes)
        found = {m.node_id: m.tier for m in matches}
        print(f"[match]  {found}")
        assert "node-b" in found and found["node-b"] == UMBPTierType.DRAM
        assert "node-c" in found and found["node-c"] == UMBPTierType.HBM

    # ------------------------------------------------------------------
    # Scenario 3 – revoke removes blocks from the index
    # ------------------------------------------------------------------
    print("\n--- Scenario 3: revoke ---")
    hashes_d = [f"evict-{i:04d}" for i in range(6)]
    to_revoke = hashes_d[:3]
    to_keep = hashes_d[3:]

    with registered_client(master_address, "node-d", DRAM_CAPS) as cd:
        cd.report_external_kv_blocks("node-d", hashes_d, UMBPTierType.DRAM)
        cd.revoke_external_kv_blocks("node-d", to_revoke)
        print(f"[node-d] revoked {len(to_revoke)} blocks, kept {len(to_keep)}")

        kept_matches = {
            h
            for m in cd.match_external_kv(to_keep)
            if m.node_id == "node-d"
            for h in m.matched_hashes
        }
        revoked_matches = {
            h
            for m in cd.match_external_kv(to_revoke)
            if m.node_id == "node-d"
            for h in m.matched_hashes
        }
        assert kept_matches == set(to_keep), f"kept mismatch: {kept_matches}"
        assert revoked_matches == set(), f"revoked still present: {revoked_matches}"
        print(
            f"[check]  kept={len(kept_matches)} blocks visible, revoked=0 blocks visible  ✓"
        )

    # ------------------------------------------------------------------
    # Scenario 4 – query with no server match returns empty list
    # ------------------------------------------------------------------
    print("\n--- Scenario 4: unknown hashes return empty list ---")
    ghost_client = UMBPMasterClient(master_address)
    result = ghost_client.match_external_kv(["no-such-hash-0", "no-such-hash-1"])
    assert result == [], f"expected [], got {result}"
    print("[check]  empty result for unknown hashes  ✓")

    print("\nAll scenarios passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with master_server() as addr:
        run_demo(addr)
