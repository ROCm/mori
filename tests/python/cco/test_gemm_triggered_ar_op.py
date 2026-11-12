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
import os
import tempfile
import time
import traceback

import pytest
import torch

N = 7168  # features
NAN_BF16 = float("nan")


def _require_gpus(world_size: int) -> None:
    n = torch.cuda.device_count()
    if n < world_size:
        pytest.skip(f"Need {world_size} GPUs, only {n} available")


class _Harness:
    """Per-rank driver: the op, the test producer and the reference."""

    def __init__(self, rank, world_size, uid_path, *, mt0, mt1, m_max, **op_kw):
        from mori.cco import Communicator, UniqueId
        from mori.jit.core import compile_genco
        from mori.jit.hip_driver import HipModule
        from mori.ops.gemm_ar import GateProducerSpec, GemmTriggeredAllReduceOp

        torch.cuda.set_device(rank)
        with open(uid_path, "rb") as f:
            uid = UniqueId.from_bytes(f.read())
        self.comm = Communicator.init(world_size, rank, uid, per_rank_vmm=2 << 30)
        self.rank, self.ws = rank, world_size
        self.mt0, self.mt1 = mt0, mt1
        self.op = GemmTriggeredAllReduceOp(
            self.comm,
            n=N,
            m_max=m_max,
            producer=GateProducerSpec(mt0=mt0, mt1=mt1),
            **op_kw,
        )
        mod = HipModule(
            compile_genco(
                "trigger_producer", os.path.dirname(os.path.abspath(__file__))
            )
        )
        self._mod = mod
        self.produce = mod.get_function("TriggerProducer")
        self.reference = mod.get_function("TriggerReference")
        self.spin = mod.get_function("TriggerSpin")
        self.ref = torch.empty(m_max * N, dtype=torch.bfloat16, device="cuda")
        self.seeds = torch.empty(world_size, dtype=torch.int64, device="cuda")
        self.calls = 0

    def seed(self, rank, call):
        return (call << 8) | rank | (1 << 40)

    def call(self, m, *, mode=0, spin_cycles=0, clobber=False, armed=False, begin=True):
        """One call; returns whether the output equals the reference exactly."""
        op, stream = self.op, torch.cuda.current_stream()
        s = stream.cuda_stream
        call = self.calls
        self.calls += 1
        if not armed:
            op.arm(m)
            if begin:
                op.begin(m)
        out = op.output_view(m)
        out.fill_(NAN_BF16)
        tiles_x = (N + self.mt0 - 1) // self.mt0
        self.produce.launch(
            (tiles_x, m // self.mt1, 1),
            (256, 1, 1),
            0,
            s,
            op.input_view(m).data_ptr(),
            N,
            self.mt0,
            self.mt1,
            op.gate_view,
            0,
            self.seed(self.rank, call),
            mode,
        )
        if spin_cycles:
            self.spin.launch((1, 1, 1), (64, 1, 1), 0, s, spin_cycles)
        op.complete(m)
        if clobber:
            got = out.clone()
            out.fill_(NAN_BF16)
        else:
            got = out
        self.seeds.copy_(
            torch.tensor(
                [self.seed(r, call) for r in range(self.ws)], dtype=torch.int64
            )
        )
        self.reference.launch(
            (1024, 1, 1),
            (256, 1, 1),
            0,
            s,
            self.ref.data_ptr(),
            N,
            m,
            self.seeds.data_ptr(),
            self.ws,
        )
        torch.cuda.synchronize()
        return torch.equal(got.reshape(-1), self.ref[: m * N])

    def close(self):
        self.op.close()
        self.comm.destroy()


def _case_exact(h, ms, calls_per_m):
    bad = []
    for m in ms:
        for _ in range(calls_per_m):
            if not h.call(m):
                bad.append(m)
    assert not bad, f"inexact calls at m={bad}"
    assert h.op.error() == 0
    if h.rank == 0:
        print(f"[rank 0] {h.calls} exact calls, m in {sorted(set(ms))}", flush=True)


def _case_rebase(h, ms, calls):
    h.op.rebase_at = h.op.layout.publishers * 5  # a re-base before every 5th call
    bad = [i for i in range(calls) if not h.call(ms[i % len(ms)])]
    assert not bad, f"inexact calls {bad}"
    assert h.op.rebases == (calls - 1) // 4, f"{h.op.rebases} re-bases"


def _case_missing_add(h, m, polls):
    assert h.call(m), "warm-up call inexact"  # also compiles the kernels everywhere
    h.comm.barrier()
    h.op.wait_polls = polls
    assert h.call(m), "a complete call failed under the short wait bound"
    h.comm.barrier()
    faulted = False
    try:
        h.call(m, mode=1 if h.rank == 0 else 0)
        h.op.check()
    except RuntimeError as e:
        faulted = "faulted" in str(e)
    assert faulted, "a missing gate add did not fault the op"
    with pytest.raises(RuntimeError):
        h.op.arm(m)


def _case_slow_reader(h, m, calls):
    bad = []
    for i in range(calls):
        slow = (i % h.ws) == h.rank
        if not h.call(m, spin_cycles=20_000_000 if slow else 0):
            bad.append(i)
    assert not bad, f"inexact calls {bad} with a slow consumer"


def _case_clobber(h, m, calls):
    bad = [i for i in range(calls) if not h.call(m, clobber=True)]
    assert not bad, f"inexact calls {bad} with the output overwritten after complete"


def _case_self_test(h, m):
    h.op.self_test()
    assert h.call(m)


def _case_misuse(h, m):
    from mori.ops.gemm_ar import GateProducerSpec, GemmTriggeredAllReduceOp

    op = h.op
    for bad, why in (
        (dict(reducer_blocks=0), "reducer_blocks"),
        (dict(reduce_mode="beside"), "reduce_mode"),
        (dict(reduce_mode="concurrent", reducer_blocks=10_000), "quarter"),
    ):
        with pytest.raises(ValueError, match=why):
            GemmTriggeredAllReduceOp(
                h.comm,
                n=N,
                m_max=m,
                producer=GateProducerSpec(mt0=h.mt0, mt1=h.mt1),
                **bad,
            )
    with pytest.raises(RuntimeError, match="armed"):
        op.complete(m)
    with pytest.raises(RuntimeError, match="armed"):
        op.begin(m)
    op.arm(m)
    with pytest.raises(RuntimeError, match="not completed"):
        op.arm(m)
    if op.reduce_mode == "concurrent":
        with pytest.raises(RuntimeError, match="begin"):
            op.complete(m)
    op.begin(m)
    with pytest.raises(RuntimeError, match="already"):
        op.begin(m)
    assert h.call(m, armed=True), "the call after the refused requests is inexact"


def _case_no_begin(h, ms, calls_per_m):
    bad = [m for m in ms for _ in range(calls_per_m) if not h.call(m, begin=False)]
    assert not bad, f"inexact calls without begin() at m={bad}"


CASES = {
    "exact": _case_exact,
    "rebase": _case_rebase,
    "missing_add": _case_missing_add,
    "slow_reader": _case_slow_reader,
    "clobber": _case_clobber,
    "self_test": _case_self_test,
    "misuse": _case_misuse,
    "no_begin": _case_no_begin,
}


def _worker(rank, world_size, uid_path, results_path, case, setup, args):
    err = None
    try:
        h = _Harness(rank, world_size, uid_path, **setup)
        try:
            CASES[case](h, **args)
        finally:
            h.close()
    except Exception:
        err = traceback.format_exc()
    with open(results_path % rank, "w") as f:
        f.write(err or "")


def _spawn_entry(rank, *args):
    _worker(rank, *args)


def _require_cus(setup) -> None:
    if setup.get("reduce_mode") != "concurrent":
        return
    need = 4 * setup.get("reducer_blocks", 32)
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    if cus < need:
        pytest.skip(f"a concurrent reducer of this size needs {need} CUs, have {cus}")


def _run(world_size, case, setup, args):
    _require_gpus(world_size)
    _require_cus(setup)
    from mori.cco import Communicator

    os.environ["MORI_ENABLE_SDMA"] = "1"
    os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")
    tmp = tempfile.mkdtemp(prefix="cco_trigger_ar_")
    uid_path = os.path.join(tmp, "uid.bin")
    results_path = os.path.join(tmp, "result_%d.txt")
    with open(uid_path, "wb") as f:
        f.write(bytes(Communicator.get_unique_id()))
    ctx = torch.multiprocessing.spawn(
        _spawn_entry,
        args=(world_size, uid_path, results_path, case, setup, args),
        nprocs=world_size,
        join=False,
    )
    # A rank that fails early leaves the others in a collective: bound the wait.
    deadline = time.monotonic() + float(
        os.environ.get("MORI_TRIGGER_TEST_TIMEOUT", "900")
    )
    while not ctx.join(timeout=5):
        if time.monotonic() > deadline:
            for p in ctx.processes:
                if p.is_alive():
                    p.kill()
            pytest.fail(f"{case} at {world_size} ranks did not finish in time")
    errs = []
    for rank in range(world_size):
        with open(results_path % rank) as f:
            e = f.read()
        if e:
            errs.append(f"rank {rank}:\n{e}")
    assert not errs, "\n\n".join(errs)


MODES = ["after", "concurrent"]


@pytest.mark.parametrize("reduce_mode", MODES)
@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("mt0,mt1", [(96, 128), (256, 256)])
def test_exact(world_size, mt0, mt1, reduce_mode):
    step = mt1 * world_size
    ms = sorted({step, 2048 // step * step or step, 8192})
    _run(
        world_size,
        "exact",
        dict(mt0=mt0, mt1=mt1, m_max=8192, reduce_mode=reduce_mode),
        dict(ms=ms, calls_per_m=3),
    )


@pytest.mark.parametrize("reduce_mode", MODES)
@pytest.mark.parametrize("world_size", [2, 8])
def test_varying_m(world_size, reduce_mode):
    step = 128 * world_size
    ms = [step * 2, step * 8, step, step * 4, step * 8, step]
    _run(
        world_size,
        "exact",
        dict(mt0=96, mt1=128, m_max=step * 8, reduce_mode=reduce_mode),
        dict(ms=ms, calls_per_m=1),
    )


@pytest.mark.parametrize("reducer_blocks", [1, 64])
def test_concurrent_reducer_sizes(reducer_blocks):
    _run(
        8,
        "exact",
        dict(
            mt0=256,
            mt1=256,
            m_max=8192,
            reduce_mode="concurrent",
            reducer_blocks=reducer_blocks,
        ),
        dict(ms=[2048, 8192], calls_per_m=3),
    )


@pytest.mark.parametrize("reduce_mode", MODES)
@pytest.mark.parametrize("ag_engine_offset", [None, 1])
def test_queue_placement(ag_engine_offset, reduce_mode):
    _run(
        8,
        "exact",
        dict(
            mt0=96,
            mt1=128,
            m_max=2048,
            ag_engine_offset=ag_engine_offset,
            reduce_mode=reduce_mode,
        ),
        dict(ms=[1024, 2048], calls_per_m=5),
    )


def test_after_mode_without_begin():
    _run(
        8,
        "no_begin",
        dict(mt0=96, mt1=128, m_max=2048),
        dict(ms=[1024, 2048], calls_per_m=3),
    )


@pytest.mark.parametrize("reduce_mode", MODES)
def test_rebase(reduce_mode):
    _run(
        8,
        "rebase",
        dict(mt0=96, mt1=128, m_max=2048, reduce_mode=reduce_mode),
        dict(ms=[1024, 2048], calls=20),
    )


@pytest.mark.parametrize("reduce_mode", MODES)
def test_missing_add_faults(reduce_mode):
    _run(
        4,
        "missing_add",
        dict(
            mt0=96, mt1=128, m_max=1024, quiesce_timeout_s=5.0, reduce_mode=reduce_mode
        ),
        # Concurrent waits start before the producer is launched.
        dict(m=1024, polls=1 << 16 if reduce_mode == "after" else 1 << 18),
    )


@pytest.mark.parametrize("reduce_mode", MODES)
def test_slow_reader(reduce_mode):
    _run(
        8,
        "slow_reader",
        dict(mt0=96, mt1=128, m_max=2048, reduce_mode=reduce_mode),
        dict(m=2048, calls=16),
    )


@pytest.mark.parametrize("reduce_mode", MODES)
def test_output_overwritten_after_complete(reduce_mode):
    _run(
        8,
        "clobber",
        dict(mt0=96, mt1=128, m_max=2048, reduce_mode=reduce_mode),
        dict(m=2048, calls=50),
    )


@pytest.mark.parametrize("reduce_mode", MODES)
def test_self_test(reduce_mode):
    _run(
        8,
        "self_test",
        dict(mt0=96, mt1=128, m_max=2048, reduce_mode=reduce_mode),
        dict(m=2048),
    )


@pytest.mark.parametrize("reduce_mode", MODES)
def test_misuse(reduce_mode):
    _run(
        2,
        "misuse",
        dict(mt0=96, mt1=128, m_max=2048, reduce_mode=reduce_mode),
        dict(m=2048),
    )


@pytest.mark.skipif(
    not os.environ.get("MORI_TRIGGER_SOAK"), reason="set MORI_TRIGGER_SOAK=calls"
)
def test_soak():
    calls = int(os.environ["MORI_TRIGGER_SOAK"])
    m = int(os.environ.get("MORI_TRIGGER_SOAK_M", "2048"))
    mode = os.environ.get("MORI_TRIGGER_SOAK_MODE", "after")
    mt1 = 256 if m >= 8192 else 128
    _run(
        8,
        "exact",
        dict(mt0=96 if mt1 == 128 else 256, mt1=mt1, m_max=m, reduce_mode=mode),
        dict(ms=[m], calls_per_m=calls),
    )
