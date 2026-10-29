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
"""Simulated MORI-IO backends (MORI_IO_SIM_ENABLE=1): no GPU or NIC is needed.

Memory pointers are fake and never dereferenced; control flow, timing and
notifications are real.
"""
import os
import subprocess
import sys
import time

import pytest

import mori.io
from mori.cpp import io_sim_profile
from mori.io import (
    BackendType,
    EngineDesc,
    IOEngine,
    IOEngineConfig,
    MemoryDesc,
    MemoryLocationType,
    PollCqMode,
    RdmaBackendConfig,
    StatusCode,
)

MiB = 1 << 20
FAKE_PTR = 0x7F0000000000
_counter = [0]


# Fixed profiles, so the timing checks do not move with the calibrated defaults.
PROFILES = {
    "RDMA": {"BW_GBPS": 45, "LATENCY_US": 5, "SEGMENT_NS": 20, "OP_US": 0},
    "TCP": {"BW_GBPS": 10, "LATENCY_US": 30, "SEGMENT_NS": 200, "OP_US": 0},
    "XGMI": {"BW_GBPS": 50, "LATENCY_US": 10, "SEGMENT_NS": 0, "OP_US": 0},
}


@pytest.fixture(autouse=True)
def sim_env(monkeypatch):
    monkeypatch.setenv("MORI_IO_SIM_ENABLE", "1")
    for k in (
        "MORI_NODE_ID",
        "MORI_IO_SIM_BACKEND",
        "MORI_IO_SIM_TIME_SCALE",
        "MORI_IO_SIM_CATALOG",
        "MORI_IO_SIM_GPU",
        "MORI_IO_SIM_NIC",
    ):
        monkeypatch.delenv(k, raising=False)
    for backend, profile in PROFILES.items():
        for k, v in profile.items():
            monkeypatch.setenv(f"MORI_IO_SIM_{backend}_{k}", str(v))
    return monkeypatch


def make_engine(tag, *, notification=True, backend=BackendType.RDMA):
    _counter[0] += 1
    eng = IOEngine(f"{tag}-{_counter[0]}", IOEngineConfig(host="127.0.0.1", port=0))
    if backend == BackendType.RDMA:
        # Same positional config sglang's mori connector passes.
        eng.create_backend(
            BackendType.RDMA,
            RdmaBackendConfig(4, -1, 4, PollCqMode.POLLING, notification),
        )
    else:
        eng.create_backend(backend)
    return eng


def connect(a, b):
    a.register_remote_engine(b.get_engine_desc())
    b.register_remote_engine(a.get_engine_desc())


def gpu_mem(eng, size, device=0):
    return eng.register_memory(FAKE_PTR, size, device, MemoryLocationType.GPU)


def write(a, src, dst, sizes, local_offs=None, remote_offs=None):
    uid = a.allocate_transfer_uid()
    local_offs = local_offs if local_offs is not None else [0] * len(sizes)
    remote_offs = remote_offs if remote_offs is not None else [0] * len(sizes)
    return (
        uid,
        a.batch_write([src], [local_offs], [dst], [remote_offs], [sizes], [uid])[0],
    )


def timed_wait(a, statuses, t0):
    rc = a.wait_all(statuses)
    return rc, (time.perf_counter() - t0) * 1e3


def model_ms(nbytes, gbps, lat_us, segs=1, seg_ns=0.0, op_us=0.0):
    return (
        nbytes / (gbps * 1e9) * 1e3 + lat_us / 1e3 + segs * seg_ns / 1e6 + op_us / 1e3
    )


def assert_close_ms(got, want, tol=0.25, slack_ms=1.5):
    assert want * (1 - tol) <= got <= want * (1 + tol) + slack_ms, (got, want)


def pop_inbound(eng, remote_key, uid, timeout_s=0.1):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = eng.pop_inbound_transfer_status(remote_key, uid)
        if st is not None:
            return st
        time.sleep(0.001)
    return None


def test_rdma_write_timing_and_notification():
    a, b = make_engine("pf"), make_engine("dc")
    connect(a, b)
    assert a.get_engine_desc().port > 0
    src, dst = gpu_mem(a, 1 << 30), gpu_mem(b, 1 << 30)
    t0 = time.perf_counter()
    uid, st = write(a, src, dst, [256 * MiB])
    rc, ms = timed_wait(a, [st], t0)
    assert rc == StatusCode.SUCCESS, st.Message()
    assert_close_ms(ms, model_ms(256 * MiB, 45, 5, 1, 20))
    inbound = pop_inbound(b, a.get_engine_desc().key, uid)
    assert inbound is not None and inbound.Succeeded()
    assert b.pop_inbound_transfer_status(a.get_engine_desc().key, uid + 1000) is None


def test_link_is_shared_fifo_per_direction():
    a, b = make_engine("pf"), make_engine("dc")
    connect(a, b)
    src, dst = gpu_mem(a, 1 << 30), gpu_mem(b, 1 << 30)
    t0 = time.perf_counter()
    sts = [write(a, src, dst, [64 * MiB])[1] for _ in range(8)]
    rc, ms = timed_wait(a, sts, t0)
    assert rc == StatusCode.SUCCESS
    assert_close_ms(ms, model_ms(8 * 64 * MiB, 45, 5, 8, 20))

    t0 = time.perf_counter()
    _, w = write(a, src, dst, [128 * MiB])
    r = a.batch_read(
        [src], [[0]], [dst], [[0]], [[128 * MiB]], [a.allocate_transfer_uid()]
    )[0]
    rc, ms = timed_wait(a, [w, r], t0)
    assert rc == StatusCode.SUCCESS
    assert_close_ms(ms, model_ms(128 * MiB, 45, 5, 1, 20))


def test_notification_disabled():
    a, b = make_engine("pf", notification=False), make_engine("dc", notification=False)
    connect(a, b)
    src, dst = gpu_mem(a, 64 * MiB), gpu_mem(b, 64 * MiB)
    uid, st = write(a, src, dst, [4096] * 64)
    assert a.wait_all([st]) == StatusCode.SUCCESS
    assert pop_inbound(b, a.get_engine_desc().key, uid, timeout_s=0.01) is None


def test_session_api():
    a, b = make_engine("pf"), make_engine("dc")
    connect(a, b)
    sess = a.create_session(gpu_mem(a, 64 * MiB), gpu_mem(b, 64 * MiB))
    assert sess is not None and sess.alive()
    st = sess.batch_write(
        [0, 4096], [8192, 0], [4096, 4096], sess.allocate_transfer_uid()
    )
    st.Wait()
    assert st.Succeeded(), st.Message()
    st = sess.read(0, 0, 1 * MiB, sess.allocate_transfer_uid())
    st.Wait()
    assert st.Succeeded(), st.Message()


def test_out_of_bounds_rejected():
    a, b = make_engine("pf"), make_engine("dc")
    connect(a, b)
    _, st = write(a, gpu_mem(a, MiB), gpu_mem(b, MiB), [4096], remote_offs=[MiB - 100])
    assert st.Code() == StatusCode.ERR_INVALID_ARGS


def test_unknown_remote_memory_raises():
    a, b = make_engine("pf"), make_engine("dc")
    connect(a, b)
    src, dst = gpu_mem(a, MiB), gpu_mem(b, MiB)
    b.deregister_memory(dst)
    with pytest.raises(RuntimeError, match="size mismatch"):
        write(a, src, dst, [4096])


def test_rdma_request_remapped_to_tcp(sim_env):
    sim_env.setenv("MORI_IO_SIM_BACKEND", "tcp")
    a, b = make_engine("pf"), make_engine("dc")
    connect(a, b)
    assert a.get_engine_desc().port > 0
    t0 = time.perf_counter()
    _, st = write(a, gpu_mem(a, 1 << 30), gpu_mem(b, 1 << 30), [256 * MiB])
    rc, ms = timed_wait(a, [st], t0)
    assert rc == StatusCode.SUCCESS
    assert_close_ms(ms, model_ms(256 * MiB, 10, 30, 1, 200))


def test_explicit_tcp_backend_notifies():
    a = make_engine("pf", backend=BackendType.TCP)
    b = make_engine("dc", backend=BackendType.TCP)
    connect(a, b)
    uid, st = write(a, gpu_mem(a, 1 << 30), gpu_mem(b, 1 << 30), [64 * MiB])
    assert a.wait_all([st]) == StatusCode.SUCCESS
    assert pop_inbound(b, a.get_engine_desc().key, uid) is not None


def test_rdma_and_tcp_cannot_share_an_engine():
    a = make_engine("pf")
    with pytest.raises(RuntimeError, match="MORI_IO_SIM_BACKEND"):
        a.create_backend(BackendType.TCP)


def test_xgmi_only_for_same_node_gpu_pairs(sim_env):
    sim_env.setenv("MORI_DISABLE_AUTO_XGMI", "0")
    sim_env.setenv("MORI_IO_SIM_XGMI_BW_GBPS", "100")
    sim_env.setenv("MORI_NODE_ID", "nodeA")
    a, b = make_engine("pf"), make_engine("dc")
    sim_env.setenv("MORI_NODE_ID", "nodeB")
    c = make_engine("dc")
    connect(a, b)
    connect(a, c)
    src = gpu_mem(a, 1 << 30)

    t0 = time.perf_counter()
    uid, st = write(a, src, gpu_mem(b, 1 << 30, device=1), [256 * MiB])
    rc, ms = timed_wait(a, [st], t0)
    assert rc == StatusCode.SUCCESS
    assert_close_ms(ms, model_ms(256 * MiB, 100, 10))
    assert pop_inbound(b, a.get_engine_desc().key, uid, timeout_s=0.01) is None

    for dst in (
        gpu_mem(c, 1 << 30),
        b.register_memory(FAKE_PTR, 1 << 30, -1, MemoryLocationType.CPU),
    ):
        t0 = time.perf_counter()
        _, st = write(a, src, dst, [256 * MiB])
        rc, ms = timed_wait(a, [st], t0)
        assert rc == StatusCode.SUCCESS
        assert_close_ms(ms, model_ms(256 * MiB, 45, 5, 1, 20))


def test_time_scale(sim_env):
    sim_env.setenv("MORI_IO_SIM_TIME_SCALE", "10")
    a, b = make_engine("pf"), make_engine("dc")
    connect(a, b)
    t0 = time.perf_counter()
    _, st = write(a, gpu_mem(a, 1 << 30), gpu_mem(b, 1 << 30), [512 * MiB])
    rc, ms = timed_wait(a, [st], t0)
    assert rc == StatusCode.SUCCESS
    assert_close_ms(ms, model_ms(512 * MiB, 45, 5, 1, 20) / 10)


def test_peer_death_fails_transfers_and_reports():
    child = subprocess.Popen(
        [sys.executable, __file__, "--target"], stdout=subprocess.PIPE, text=True
    )
    try:
        edesc = EngineDesc.unpack(bytes.fromhex(child.stdout.readline().strip()))
        mdesc = MemoryDesc.unpack(bytes.fromhex(child.stdout.readline().strip()))
        a = make_engine("pf")
        a.register_remote_engine(edesc)
        src = gpu_mem(a, 64 * MiB)
        _, st = write(a, src, mdesc, [MiB])
        assert a.wait_all([st]) == StatusCode.SUCCESS, st.Message()
    finally:
        child.kill()
        child.wait()
    time.sleep(0.05)
    _, st = write(a, src, mdesc, [MiB])
    assert a.wait_all([st]) == StatusCode.ERR_RDMA_OP
    ev = a.pop_peer_failure()
    assert ev is not None and ev.remote_engine_key == edesc.key
    _, st = write(a, src, mdesc, [MiB])
    assert st.Failed()


def use_catalog(sim_env, tmp_path, *texts):
    """Catalog files in order, with the fixture's per-value variables cleared."""
    for backend, profile in PROFILES.items():
        for k in profile:
            sim_env.delenv(f"MORI_IO_SIM_{backend}_{k}")
    paths = []
    for i, text in enumerate(texts):
        paths.append(tmp_path / f"catalog{i}.yaml")
        paths[-1].write_text(text)
    sim_env.setenv("MORI_IO_SIM_CATALOG", ",".join(map(str, paths)))


def entry(backend, hardware, bw, lat=1, seg=0, op=0, extra=""):
    key = "gpu" if backend == "xgmi" else "nic"
    return (
        f"- backend: {backend}\n  {key}: {hardware}\n  bandwidth_gbps: {bw}\n"
        f"  latency_us: {lat}\n  per_segment_ns: {seg}\n  per_op_us: {op}\n{extra}"
    )


def test_catalog_entry_sets_timing(sim_env, tmp_path):
    use_catalog(
        sim_env, tmp_path, entry("rdma", "testnic", 20, lat=50, seg=1000, op=30)
    )
    sim_env.setenv("MORI_IO_SIM_NIC", "testnic")
    assert io_sim_profile(BackendType.RDMA) == {
        "bw_gbps": 20,
        "latency_us": 50,
        "per_segment_ns": 1000,
        "per_op_us": 30,
    }
    a, b = make_engine("pf"), make_engine("dc")
    connect(a, b)
    t0 = time.perf_counter()
    _, st = write(a, gpu_mem(a, 1 << 30), gpu_mem(b, 1 << 30), [MiB] * 64)
    rc, ms = timed_wait(a, [st], t0)
    assert rc == StatusCode.SUCCESS
    assert_close_ms(ms, model_ms(64 * MiB, 20, 50, 64, 1000, 30))


def test_catalogs_layer_and_variables_win(sim_env, tmp_path):
    use_catalog(
        sim_env,
        tmp_path,
        entry("rdma", "testnic", 20) + entry("xgmi", "testgpu", 30),
        "# a later file replaces the rdma entry only\n" + entry("rdma", "testnic", 40),
    )
    sim_env.setenv("MORI_IO_SIM_GPU", "testgpu")
    sim_env.setenv("MORI_IO_SIM_NIC", "testnic")
    assert io_sim_profile(BackendType.RDMA)["bw_gbps"] == 40
    assert io_sim_profile(BackendType.XGMI)["bw_gbps"] == 30
    assert io_sim_profile(BackendType.TCP)["bw_gbps"] == 10  # no tcp entry: the default
    sim_env.setenv("MORI_IO_SIM_RDMA_LATENCY_US", "7")
    assert io_sim_profile(BackendType.RDMA)["latency_us"] == 7
    assert io_sim_profile(BackendType.RDMA)["bw_gbps"] == 40


@pytest.mark.parametrize(
    "text, error",
    [
        ("- backend: rdma\n  nic: n\n  bandwidth_gbps: 1\n", "missing latency_us"),
        (entry("rdma", "n", 1).replace("  nic: n\n", ""), "missing nic"),
        (
            entry("xgmi", "g", 1, extra="  nic: n\n"),
            "a xgmi entry names a gpu, not a nic",
        ),
        (entry("nvlink", "n", 1), "backend must be rdma, tcp or xgmi"),
        (entry("rdma", "n", "fast"), "bandwidth_gbps must be a number"),
        (entry("rdma", "n", 0), "bandwidth must be positive"),
        (entry("rdma", "n", 1, extra="  per_op_ns: 3\n"), "unknown field 'per_op_ns'"),
        (entry("rdma", "n", 1) + entry("rdma", "n", 2), "second rdma entry for nic n"),
    ],
)
def test_bad_catalog_is_rejected(sim_env, tmp_path, text, error):
    use_catalog(sim_env, tmp_path, text)
    with pytest.raises(RuntimeError, match=error):
        make_engine("pf")


SHIPPED_CATALOG = os.path.join(os.path.dirname(mori.io.__file__), "sim_catalog.yaml")


def test_unknown_hardware_lists_the_catalog(sim_env):
    sim_env.setenv("MORI_IO_SIM_CATALOG", SHIPPED_CATALOG)
    sim_env.setenv("MORI_IO_SIM_NIC", "nosuchnic")
    with pytest.raises(
        RuntimeError,
        match="MORI_IO_SIM_NIC=nosuchnic is not in MORI_IO_SIM_CATALOG, "
        "which has .*pollara400",
    ):
        make_engine("pf")


def test_hardware_needs_a_catalog(sim_env):
    sim_env.setenv("MORI_IO_SIM_NIC", "pollara400")
    with pytest.raises(
        RuntimeError,
        match="MORI_IO_SIM_NIC=pollara400 is not in MORI_IO_SIM_CATALOG, which has none",
    ):
        make_engine("pf")


def test_shipped_catalog(sim_env, tmp_path):
    use_catalog(sim_env, tmp_path)
    sim_env.setenv("MORI_IO_SIM_CATALOG", SHIPPED_CATALOG)
    sim_env.setenv("MORI_IO_SIM_GPU", "mi355x")
    sim_env.setenv("MORI_IO_SIM_NIC", "pollara400")
    assert io_sim_profile(BackendType.XGMI)["bw_gbps"] == 61.4
    assert io_sim_profile(BackendType.RDMA)["bw_gbps"] == 47.4


if __name__ == "__main__" and "--target" in sys.argv:
    os.environ["MORI_IO_SIM_ENABLE"] = "1"
    t = make_engine("target")
    m = gpu_mem(t, 64 * MiB)
    print(t.get_engine_desc().pack().hex(), flush=True)
    print(m.pack().hex(), flush=True)
    time.sleep(600)
