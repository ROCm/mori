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
# SPDX-License-Identifier: MIT

import ctypes
import json
import os
import shlex
import shutil
import signal

import pytest

umbp = pytest.importorskip("mori.cpp")


@pytest.mark.parametrize("mode", ["standalone", "embedded", "distributed-local"])
def test_shared_policy_and_page_size_reach_backend(mode, tmp_path, monkeypatch):
    binary = os.environ.get("UMBP_STANDALONE_BIN") or shutil.which(
        "umbp_standalone_server"
    )
    if mode == "standalone" and not binary:
        pytest.skip("umbp_standalone_server executable is required")
    # Isolate the child from any deployment configured in the test environment.
    for name in list(os.environ):
        if name.startswith("UMBP_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("UMBP_STANDALONE_SHM_DIR", str(tmp_path))

    policy_path = tmp_path / "policy.json"
    ssd_dir = tmp_path / "ssd"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entry_tier": "hot",
                "backends": {
                    "dram": {"type": "dram", "capacity": "32KiB"},
                    "ssd": {
                        "type": "ssd",
                        "capacity": "1MiB",
                        "path": str(ssd_dir),
                    },
                },
                "tiers": [
                    {
                        "name": "hot",
                        "backends": {"dram": 1},
                        "offload_to": ["cold"],
                        "offload_trigger": "watermark",
                    },
                    {"name": "cold", "backends": {"ssd": 1}},
                ],
            }
        )
    )
    # Record the auto-started process so the test can stop only its own server.
    pid_path = tmp_path / "server.pid"
    wrapper = tmp_path / "server-wrapper"
    if mode == "standalone":
        wrapper.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "$$" > {shlex.quote(str(pid_path))}\n'
            f'exec {shlex.quote(binary)} "$@"\n'
        )
        wrapper.chmod(0o700)
        monkeypatch.setenv("UMBP_STANDALONE_BIN", str(wrapper))

    config = umbp.UMBPConfig()
    config.dram.capacity_bytes = 1 << 20
    config.ssd.enabled = False  # The policy owns the server's SSD backend.
    config.backend_policy_path = str(policy_path)
    # The 32 KiB DRAM backend cannot fit the default 2 MiB page.
    config.page_size = 4096
    if mode == "standalone":
        standalone = umbp.UMBPStandaloneProcessConfig()
        standalone.address = f"unix://{tmp_path}/server.sock"
        standalone.auto_start = True
        standalone.startup_timeout_ms = 10000
        config.standalone_process = standalone
    elif mode == "distributed-local":
        distributed = umbp.UMBPDistributedConfig()
        distributed.master_config.node_id = "shared-config-test"
        distributed.master_config.node_address = "127.0.0.1"
        config.distributed = distributed

    client = None
    allocator = umbp.UMBPHostMemAllocator()
    buffer = allocator.alloc(4096, umbp.UMBPHostBufferBacking.AnonymousShm)
    try:
        assert buffer
        client = umbp.UMBPClient(config)
        assert ssd_dir.is_dir()
        assert client.register_memory(buffer.ptr, buffer.mapped_size)
        payload = bytes(range(256)) * 16
        ctypes.memmove(buffer.ptr, payload, len(payload))
        assert client.put_from_ptr("test-key", buffer.ptr, len(payload))
        ctypes.memset(buffer.ptr, 0, len(payload))
        assert client.get_into_ptr("test-key", buffer.ptr, len(payload))
        assert ctypes.string_at(buffer.ptr, len(payload)) == payload
    finally:
        client = None  # The Python binding closes the client on destruction.
        allocator.free(buffer)
        if pid_path.exists():
            pid = int(pid_path.read_text())
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
