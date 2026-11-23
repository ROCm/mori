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
"""Standalone registration ownership must survive overlapping container PIDs."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

WORKER = r"""
import ctypes
import os
from pathlib import Path
import sys
from mori.cpp import (
    UMBPClient, UMBPConfig, UMBPStandaloneProcessConfig,
    UMBPHostMemAllocator, UMBPHostBufferBacking,
)

root, role = Path(sys.argv[1]), sys.argv[2]
assert os.getpid() == 1
config = UMBPConfig()
standalone = UMBPStandaloneProcessConfig()
standalone.address = 'unix://' + str(root / 'server.sock')
standalone.auto_start = False
standalone.startup_timeout_ms = 10000
config.standalone_process = standalone
allocator = UMBPHostMemAllocator()
buffer = allocator.alloc(1 << 20, UMBPHostBufferBacking.AnonymousShm, prefault=False)
client = UMBPClient(config)
assert client.register_memory(buffer.ptr, 1 << 20)
value = ord(role)
ctypes.memset(buffer.ptr, value, 4096)
assert list(client.batch_put_ranges_from_ptr(
    [role], [4096], [[buffer.ptr]], [[4096]], [[0]])) == [True]
assert client.flush()

def restore():
    ctypes.memset(buffer.ptr, 0, 4096)
    assert list(client.batch_get_ranges_into_ptr(
        [role], [[buffer.ptr]], [[4096]], [[0]])) == [True]
    assert ctypes.string_at(buffer.ptr, 4096) == bytes([value]) * 4096

restore()
(root / (role + '.ready')).touch()
assert sys.stdin.readline().strip() == 'continue'
if role == 'b':
    restore()
    client.deregister_memory(buffer.ptr)
    assert client.register_memory(buffer.ptr, 1 << 20)
    restore()
client.deregister_memory(buffer.ptr)
allocator.free(buffer)
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Linux shared-memory transport")
def test_deregister_does_not_unmap_another_process_with_same_pid(tmp_path):
    mori = pytest.importorskip("mori")
    pytest.importorskip("mori.umbp")
    server_bin = Path(
        os.environ.get(
            "UMBP_STANDALONE_BIN",
            str(Path(mori.__file__).parent / "umbp_standalone_server"),
        )
    )
    if not server_bin.is_file():
        pytest.skip("standalone server was not built")
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required for the PID-namespace test shim")

    # Fresh processes reset the old process-local counter. Override only their
    # view of getpid(), as in two containers where each client is PID 1. This
    # avoids requiring Docker or privileged PID namespaces in the test runner.
    shim = tmp_path / "pid.c"
    shim.write_text("#include <unistd.h>\npid_t getpid(void) { return 1; }\n")
    library = tmp_path / "pid.so"
    subprocess.run([cc, "-shared", "-fPIC", str(shim), "-o", str(library)], check=True)
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    env = dict(os.environ)
    env.update(
        UMBP_DRAM_CAPACITY=str(64 << 20),
        UMBP_DRAM_HOST_REGISTER="0",
        UMBP_DRAM_USE_HUGEPAGES="0",
        UMBP_DRAM_PREFAULT="0",
        UMBP_STANDALONE_METRICS_PORT="0",
        UMBP_SSD_ENABLED="0",
    )
    children, logs = [], []
    try:
        log = (tmp_path / "server.log").open("w")
        logs.append(log)
        server = subprocess.Popen(
            [str(server_bin), "unix://" + str(tmp_path / "server.sock")],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        children.append(server)
        clients = []
        child_env = dict(env, LD_PRELOAD=str(library))
        for role in ("a", "b"):
            log = (tmp_path / (role + ".log")).open("w")
            logs.append(log)
            client = subprocess.Popen(
                [sys.executable, str(worker), str(tmp_path), role],
                env=child_env,
                stdin=subprocess.PIPE,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            children.append(client)
            clients.append(client)
            deadline = time.monotonic() + 30
            while not (tmp_path / (role + ".ready")).exists():
                assert client.poll() is None, (tmp_path / (role + ".log")).read_text()
                assert time.monotonic() < deadline, "client registration timed out"
                time.sleep(0.05)
        for role, client in zip(("a", "b"), clients):
            client.stdin.write("continue\n")
            client.stdin.flush()
            assert client.wait(timeout=30) == 0, (
                tmp_path / (role + ".log")
            ).read_text()
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
