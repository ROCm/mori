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
"""Check the BF16 pipeline's happens-before edges without a GPU.

Execute the actual kernel's scheduling statements with tracing loaders. Memory
operations may complete as late as their first required wait. Barrier ordinals,
not source locations, determine inter-wave ordering: wave_m=1 starts one
barrier behind wave_m=0. This catches races that a hot numerical run can hide.
The model assumes FIFO completion of G2S loads, as the vmcnt arithmetic does.
It does not model instruction latency or replace ISA/numerical validation.
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

KERNEL = (
    Path(__file__).resolve().parents[3]
    / "python/mori/ops/gemm_ag/_gemm_a16w16_8wave.py"
)


@dataclass(frozen=True)
class Position:
    wave: int
    phase: int
    order: int

    def before(self, other):
        return self.phase < other.phase or (
            self.wave == other.wave and self.order < other.order
        )


@dataclass
class Access:
    buffer: str
    tile: int
    rows: range
    issued: Position
    done: Position | None = None


class Wave:
    def __init__(self, wave, bm, bn):
        self.wave = wave
        self.bm, self.bn = bm, bn
        self.phase = self.order = 0
        self.writes = []
        self.reads = []
        self.generations = {}

    def position(self):
        self.order += 1
        return Position(self.wave, self.phase, self.order)

    def barrier(self):
        self.phase += 1

    def wait_vm(self, count):
        pos = self.position()
        for write in self.writes[: len(self.writes) - count if count else None]:
            if write.done is None:
                write.done = pos
        self.barrier()

    def wait_lds(self):
        pos = self.position()
        for read in self.reads:
            if read.done is None:
                read.done = pos

    def write(self, buffer, offset):
        rows = (self.bm if buffer[0] == "a" else self.bn) // 2
        for step in range(rows // 64):
            start = self.wave * 8 + step * 64
            self.writes.append(
                Access(buffer, offset // 64, range(start, start + 8), self.position())
            )

    def read(self, buffer):
        if buffer[0] == "a":
            size, part = self.bm // 4, self.wave // 4
        else:
            size, part = self.bn // 8, self.wave % 4
        tile = self.generations.get(buffer, int(buffer[1]))
        self.generations[buffer] = tile + 2
        read = Access(
            buffer, tile, range(part * size, (part + 1) * size), self.position()
        )
        self.reads.append(read)
        return read

    def mma(self, a, b, c, set_prio=True):
        # LLVM must wait for the operand registers before their first use.
        pos = self.position()
        for read in (a, b):
            if read.done is None:
                read.done = pos
        if set_prio:
            self.barrier()
        return c


def trace_pipeline(source, bm, bn, k_iters, policy):
    tree = ast.parse(source)
    factory = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compile_bf16_gemm_ag"
    )
    kernel = next(
        n
        for n in factory.body
        if isinstance(n, ast.FunctionDef) and n.name == "kernel_gemm"
    )
    # Extract the real prologue, mainloop, tail, and closing half-wave barrier.
    start = next(
        i for i, n in enumerate(kernel.body) if ast.unparse(n).startswith("b_g2s.load(")
    )
    end = next(
        i
        for i, n in enumerate(kernel.body)
        if ast.unparse(n).startswith("store_c.store(")
    )
    code = compile(ast.Module(kernel.body[start:end], []), str(KERNEL), "exec")
    main_wait = next(
        n
        for n in factory.body
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_MAIN_WAIT"
    )
    wait_code = compile(ast.Module([main_wait], []), str(KERNEL), "exec")
    waves = []
    for wave_id in range(8):
        wave = Wave(wave_id, bm, bn)
        env = dict(
            K_ITERS=k_iters,
            split_k=1,
            const_expr=lambda value: value,
            BLOCK_K=64,
            B_K_STEP=64,
            BLOCK_M=bm,
            BLOCK_N=bn,
            N_TILES_A=bm // 64,
            N_TILES_B=bn // 128,
            LDS_BLOCK_M=bm // 2,
            LDS_BLOCK_N=bn // 2,
            N_LDS_STEPS_A=bm // 128,
            N_LDS_STEPS_B=bn // 128,
            A0_gl_offset=0,
            A1_gl_offset=0,
            B0_gl_offset=0,
            B1_gl_offset=0,
            wave_m=wave_id // 4,
            wave_n=wave_id % 4,
            block_m=0,
            block_n=0,
            c00_frag=None,
            c01_frag=None,
            c10_frag=None,
            c11_frag=None,
            range_constexpr=range,
            wait_policy=policy,
            a_cur0="a00",
            a_cur1="a01",
            a_next0="a10",
            a_next1="a11",
            b_cur0="b00",
            b_cur1="b01",
            b_next0="b10",
            b_next1="b11",
            a_g2s=SimpleNamespace(load=wave.write),
            b_g2s=SimpleNamespace(load=wave.write),
            a_s2r=SimpleNamespace(load=wave.read),
            b_s2r=SimpleNamespace(load=wave.read),
            mfma=SimpleNamespace(call=wave.mma),
            rocdl=SimpleNamespace(s_barrier=wave.barrier, s_setprio=lambda _: None),
            _wait_lds_reads=wave.wait_lds,
            wait_barrier=wave.wait_vm,
            _wb=lambda count, wave=wave: wave.wait_vm(
                0 if policy == "conservative" else count
            ),
        )
        exec(wait_code, env)  # noqa: S102 - trace trusted local kernel source
        exec(code, env)  # noqa: S102 - trace trusted local kernel source
        waves.append(wave)
    assert len({w.phase for w in waves}) == 1, "unbalanced half-wave barriers"
    return waves


def hazards(waves):
    writers = {}
    readers = {}
    for wave in waves:
        for access in wave.writes:
            for row in access.rows:
                key = access.buffer, access.tile, row
                assert key not in writers, f"multiple writers: {key}"
                writers[key] = access
        for access in wave.reads:
            for row in access.rows:
                readers.setdefault((access.buffer, access.tile, row), []).append(access)
    errors = set()
    for key, reads in readers.items():
        writer = writers[key]
        overwrite = writers.get((key[0], key[1] + 2, key[2]))
        for read in reads:
            if writer.done is None or not writer.done.before(read.issued):
                errors.add(f"RAW {key[0]} k={key[1]}")
            if overwrite and (
                read.done is None or not read.done.before(overwrite.issued)
            ):
                errors.add(f"WAR {key[0]} k={key[1]}")
    return sorted(errors)


@pytest.mark.parametrize("bm,bn", [(128, 128), (128, 256), (256, 128), (256, 256)])
@pytest.mark.parametrize("k_iters", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("policy", ["tuned", "safe", "conservative"])
def test_bf16_pipeline_orders_lds_accesses(bm, bn, k_iters, policy):
    assert not hazards(trace_pipeline(KERNEL.read_text(), bm, bn, k_iters, policy))


def test_model_detects_missing_b0_read_completion():
    # Mutation control: removing the wait must expose the original WAR even
    # with correct vmcnt thresholds and all barriers still in place.
    source = KERNEL.read_text().replace("            _wait_lds_reads()\n", "")
    errors = hazards(trace_pipeline(source, 128, 256, 4, "tuned"))
    assert any(e.startswith("WAR b") for e in errors), errors


def test_model_detects_late_tail_wait():
    # Move the drain back after a0's final S2R, as in the broken tail.
    source = KERNEL.read_text()
    start = source.index("        # Step k = K_ITERS - 2")
    head, tail = source[:start], source[start:]
    tail = tail.replace("        wait_barrier(0)\n", "        rocdl.s_barrier()\n", 1)
    penultimate, final = tail.split("        # Step k = K_ITERS - 1", 1)
    final = final.replace(
        "        a0_frag = a_s2r.load(a_cur0)\n        rocdl.s_barrier()",
        "        a0_frag = a_s2r.load(a_cur0)\n        wait_barrier(0)",
        1,
    )
    tail = penultimate + "        # Step k = K_ITERS - 1" + final
    errors = hazards(trace_pipeline(head + tail, 128, 256, 4, "tuned"))
    assert any(e.startswith(("RAW b", "RAW a")) for e in errors), errors


def test_model_detects_permissive_asymmetric_wait():
    source = KERNEL.read_text()
    source = source.replace(
        "else min(\n            2 * N_LDS_STEPS_A + N_LDS_STEPS_B,\n"
        "            N_LDS_STEPS_A + 2 * N_LDS_STEPS_B,\n        )",
        "else 2 * N_LDS_STEPS_A + N_LDS_STEPS_B",
    )
    errors = hazards(trace_pipeline(source, 256, 128, 4, "tuned"))
    assert any(e.startswith("RAW a") for e in errors), errors
