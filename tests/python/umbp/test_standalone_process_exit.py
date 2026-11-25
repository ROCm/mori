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
"""A dead worker's registration is not a cache object's lifetime."""

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

WORKER = r"""
import ctypes
from pathlib import Path
import sys
from mori.cpp import (
    UMBPClient, UMBPConfig, UMBPStandaloneProcessConfig,
    UMBPHostMemAllocator, UMBPHostBufferBacking,
)
root, role = Path(sys.argv[1]), sys.argv[2]
config = UMBPConfig()
standalone = UMBPStandaloneProcessConfig()
standalone.address = 'unix://' + str(root / 'server.sock')
standalone.auto_start = False
standalone.startup_timeout_ms = 10000
config.standalone_process = standalone
allocator = UMBPHostMemAllocator()
buffer = allocator.alloc(8 << 20, UMBPHostBufferBacking.AnonymousShm, prefault=False)
client = UMBPClient(config)
assert client.register_memory(buffer.ptr, 8 << 20)
ctypes.memset(buffer.ptr, ord(role), 4096)
assert list(client.batch_put_ranges_from_ptr(
    [role], [4096], [[buffer.ptr]], [[4096]], [[0]])) == [True]
assert client.flush()
(root / (role + '.ready')).touch()
assert sys.stdin.readline().strip() == 'continue'
for key in ('a', 'b'):
    ctypes.memset(buffer.ptr, 0, 4096)
    assert list(client.batch_get_ranges_into_ptr(
        [key], [[buffer.ptr]], [[4096]], [[0]])) == [True]
    assert ctypes.string_at(buffer.ptr, 4096) == bytes([ord(key)]) * 4096
client.deregister_memory(buffer.ptr)
assert client.register_memory(buffer.ptr, 8 << 20)
client.deregister_memory(buffer.ptr)
del client
allocator.free(buffer)
"""


def _mapped_bytes(pid):
    total = 0
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        if "/memfd:umbp_host_buffer" in line:
            start, end = line.split()[0].split("-")
            total += int(end, 16) - int(start, 16)
    return total


def _wait(predicate, process, log, seconds=10):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert process.poll() is None, log.read_text()
        assert time.monotonic() < deadline, log.read_text()
        time.sleep(0.02)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd and memfd")
def test_killed_worker_is_reclaimed_without_losing_cached_objects(tmp_path):
    mori = pytest.importorskip("mori")
    pytest.importorskip("mori.umbp")
    binary = Path(
        os.environ.get(
            "UMBP_STANDALONE_BIN",
            str(Path(mori.__file__).parent / "umbp_standalone_server"),
        )
    )
    if not binary.is_file():
        pytest.skip("standalone server was not built")
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    env = dict(
        os.environ,
        UMBP_DRAM_CAPACITY=str(64 << 20),
        UMBP_DRAM_HOST_REGISTER="0",
        UMBP_DRAM_USE_HUGEPAGES="0",
        UMBP_DRAM_PREFAULT="0",
        UMBP_STANDALONE_METRICS_PORT="0",
        UMBP_SSD_ENABLED="0",
    )
    children, logs = [], []
    try:
        server_log = tmp_path / "server.log"
        log = server_log.open("w")
        logs.append(log)
        server = subprocess.Popen(
            [str(binary), "unix://" + str(tmp_path / "server.sock")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        children.append(server)
        clients = []
        for role in ("a", "b"):
            path = tmp_path / (role + ".log")
            log = path.open("w")
            logs.append(log)
            client = subprocess.Popen(
                [sys.executable, str(worker), str(tmp_path), role],
                env=env,
                stdin=subprocess.PIPE,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            children.append(client)
            clients.append(client)
            _wait(lambda: (tmp_path / (role + ".ready")).exists(), client, path, 30)
        assert _mapped_bytes(server.pid) == 16 << 20
        clients[0].kill()
        assert clients[0].wait(timeout=10) == -9
        _wait(lambda: _mapped_bytes(server.pid) == 8 << 20, server, server_log)
        clients[1].stdin.write("continue\n")
        clients[1].stdin.flush()
        assert clients[1].wait(timeout=10) == 0, (tmp_path / "b.log").read_text()
        _wait(lambda: _mapped_bytes(server.pid) == 0, server, server_log)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
        for log in logs:
            log.close()
