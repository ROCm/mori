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
"""Key-space and payload logic of umbp_prim_bench.

All of mori's imports in that module are function-local, so the parts tested
here -- which keys a call names, what each one must contain, and what answer
each is expected to give -- run without a store, a GPU or a server.
"""
import pytest

bench = pytest.importorskip("umbp_prim_bench")


def _cfg(**over):
    base = dict(
        backend="umbp-server",
        address="unix:///tmp/x.grpc.sock",
        ops=("exists",),
        rank_counts=(1,),
        keys_per_rank=8,
        batch=8,
        passes=1,
        ranges=4,
        range_bytes=16,
        loc="host",
        register=False,
        verify=True,
        key_prefix="t",
        dram_capacity=1 << 30,
        exists_absent_frac=0.0,
    )
    base.update(over)
    return bench.Config(**base)


# --- payload ---------------------------------------------------------------


def test_payload_is_deterministic_key_derived_and_full_length():
    a = bench._payload("k1", 4096)
    assert len(a) == 4096
    assert a == bench._payload("k1", 4096), "must reproduce across calls"
    assert a != bench._payload("k2", 4096), "different keys, different bytes"


def test_payload_is_offset_sensitive():
    """A repeated-tag pattern would verify clean against a shifted read; this
    is what makes a wrong range offset detectable at all."""
    p = bench._payload("k1", 4096)
    assert p[0:64] != p[64:128]
    assert p[0:64] != p[32:96], "a 32-byte shift must change the bytes"


def test_truncated_or_zero_filled_reads_are_detectable():
    """The keystream is a stream, so a short payload IS a prefix of a long
    one -- which is harmless only because verification always compares the
    full object length.  These are the shapes that must not compare equal.
    """
    full = bench._payload("k", 128)
    assert full[:64] + bytes(64) != full, "a half-filled object must differ"
    assert bytes(128) != full, "an untouched buffer must differ"


# --- key spaces ------------------------------------------------------------


def test_absent_keys_are_disjoint_from_every_written_key():
    cfg = _cfg()
    present = set()
    for op in ("put", "get", "exists", "put_ranges", "get_ranges"):
        for gen in range(0, 4):
            present |= set(cfg.keys(op, 1, 0, gen))
    absent = set(cfg.keys("exists", 1, 0, 0, absent=True))
    assert absent and not (absent & present)


def test_absent_keys_have_the_same_shape_as_present_ones():
    """Same length and prefix, so mixing the two does not also mix two
    different string-hashing costs into the measurement."""
    cfg = _cfg()
    present = cfg.keys("exists", 1, 0, 0)
    absent = cfg.keys("exists", 1, 0, 0, absent=True)
    assert {len(k) for k in present} == {len(k) for k in absent}


# --- absent-key mixing -----------------------------------------------------


def _exists_op(frac, keys_per_rank=8, batch=8):
    cfg = _cfg(exists_absent_frac=frac, keys_per_rank=keys_per_rank, batch=batch)
    return cfg, bench.ExistsOp(cfg, client=None, buf=None, ranks=1, rank=0)


@pytest.mark.parametrize("frac,expected", [(0.0, 0), (0.25, 2), (0.5, 4), (1.0, 8)])
def test_absent_count_matches_the_requested_fraction(frac, expected):
    _, op = _exists_op(frac)
    assert len(op.absent_slots(8)) == expected


def test_absent_slots_are_spread_not_blocked():
    """Blocked at one end, whole calls would be single-population and the
    per-call latency would depend on which chunk you looked at."""
    _, op = _exists_op(0.5)
    slots = sorted(op.absent_slots(8))
    assert slots == [0, 2, 4, 6]


def test_expectation_tracks_the_absent_slots():
    cfg, op = _exists_op(0.5)
    (batch,) = op.prepare(0)
    absent = set(cfg.keys("exists", 1, 0, 0, absent=True))
    assert batch.expect is not None
    for name, exp in zip(batch.key_names, batch.expect):
        assert exp == (name not in absent), name
    assert sum(batch.expect) == 4


def test_all_present_probe_carries_no_absent_keys():
    cfg, op = _exists_op(0.0)
    (batch,) = op.prepare(0)
    assert all(batch.expect)
    assert set(batch.key_names) <= set(cfg.keys("exists", 1, 0, 0))


def test_all_absent_probe_expects_every_key_missing():
    cfg, op = _exists_op(1.0)
    (batch,) = op.prepare(0)
    assert not any(batch.expect)
    assert set(batch.key_names) <= set(cfg.keys("exists", 1, 0, 0, absent=True))


def test_absent_keys_do_not_shrink_the_call():
    """The probe must stay the same size at every fraction, or the sweep
    compares calls of different widths."""
    for frac in (0.0, 0.5, 1.0):
        _, op = _exists_op(frac, keys_per_rank=16, batch=4)
        batches = op.prepare(0)
        assert [b.keys for b in batches] == [4, 4, 4, 4]
        assert all(len(b.key_names) == 4 for b in batches)


def test_fraction_is_rejected_outside_zero_to_one():
    for bad in ("-0.1", "1.5"):
        with pytest.raises(SystemExit):
            bench.parse_args(["umbp-local", "--exists-absent-frac", bad])
