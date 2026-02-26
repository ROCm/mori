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
import time

import torch
from tests.python.utils import get_free_port

from mori.io import (
    BackendType,
    IOEngine,
    IOEngineConfig,
    MemoryDesc,
    TcpBackendConfig,
    set_log_level,
)


def _wait_inbound_status(engine, remote_engine_key, remote_transfer_uid, timeout_s=5.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = engine.pop_inbound_transfer_status(remote_engine_key, remote_transfer_uid)
        if st is not None:
            return st
        time.sleep(0.001)
    raise RuntimeError("Timed out waiting for inbound status")


def _create_tcp_engine_pair(name_prefix, port_a=0, port_b=0):
    cfg_a = IOEngineConfig(host="127.0.0.1", port=port_a)
    cfg_b = IOEngineConfig(host="127.0.0.1", port=port_b)

    a = IOEngine(key=f"{name_prefix}_a", config=cfg_a)
    b = IOEngine(key=f"{name_prefix}_b", config=cfg_b)

    a.create_backend(BackendType.TCP, TcpBackendConfig())
    b.create_backend(BackendType.TCP, TcpBackendConfig())

    a_desc = a.get_engine_desc()
    b_desc = b.get_engine_desc()
    a.register_remote_engine(b_desc)
    b.register_remote_engine(a_desc)
    return a, b, a_desc, b_desc


def test_tcp_engine_desc_port_zero_auto_bind():
    set_log_level("error")
    engine = IOEngine(key="engine_tcp_port0", config=IOEngineConfig(host="127.0.0.1", port=0))
    engine.create_backend(BackendType.TCP, TcpBackendConfig())
    desc = engine.get_engine_desc()
    assert desc.port > 0


def test_tcp_cpu_write_read_and_batch():
    set_log_level("error")
    a, b, a_desc, b_desc = _create_tcp_engine_pair("tcp_cpu", get_free_port(), get_free_port())

    # Allocate CPU tensors and register memory.
    src = torch.arange(0, 1024 * 4, dtype=torch.uint8)
    dst = torch.zeros_like(src)
    src_md = a.register_torch_tensor(src)
    dst_md = b.register_torch_tensor(dst)

    # MemoryDesc serialization should work for TCP too.
    packed = dst_md.pack()
    dst_md_remote = MemoryDesc.unpack(packed)
    assert dst_md == dst_md_remote

    # Single write
    uid = a.allocate_transfer_uid()
    st = a.write(src_md, 0, dst_md, 0, src.numel() * src.element_size(), uid)
    st.Wait()
    assert st.Succeeded(), st.Message()
    bst = _wait_inbound_status(b, a_desc.key, uid)
    assert bst.Succeeded(), bst.Message()
    assert torch.equal(src, dst)

    # Single read (b -> a)
    dst.zero_()
    uid = a.allocate_transfer_uid()
    st = a.read(src_md, 0, dst_md, 0, src.numel() * src.element_size(), uid)
    st.Wait()
    assert st.Succeeded(), st.Message()
    bst = _wait_inbound_status(b, a_desc.key, uid)
    assert bst.Succeeded(), bst.Message()
    assert torch.equal(src, dst)

    # Batch write via session
    sess = a.create_session(src_md, dst_md)
    assert sess is not None
    offsets = [0, 256, 512, 768]
    sizes = [128, 128, 128, 128]

    dst.zero_()
    uid = sess.allocate_transfer_uid()
    st = sess.batch_write(offsets, offsets, sizes, uid)
    st.Wait()
    assert st.Succeeded(), st.Message()
    bst = _wait_inbound_status(b, a_desc.key, uid)
    assert bst.Succeeded(), bst.Message()

    for off, sz in zip(offsets, sizes):
        assert torch.equal(src[off : off + sz], dst[off : off + sz])

