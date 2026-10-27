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
"""V2 tuning tables: the CSV format, its checks, and the tuning save.

One CSV per GPU model, kernel family and EP size holds both phases. A malformed
table raises rather than being skipped: one that loaded wrong would run untuned
geometry and nothing else would say so. Every shipped table is loaded from
wherever this mori is installed, which also catches a table left out of
package_data. CPU only.
"""

import re

import pytest
import torch

from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig
from mori.ops.dispatch_combine_v2 import hip_tuning_configs as tuning

ARCH, MODEL = "gfx942", "testgpu"
NAME = re.compile(
    r"^(gfx\d+)_([a-z0-9]+)_(InterNodeV2|InterNodeV2LL|IntraNodeV2)_ep(\d+)\.csv$"
)


def shipped_tables():
    return sorted(tuning._INTERNODE_TABLE_DIR.glob("*.csv"))


def test_every_shipped_table_loads():
    assert shipped_tables(), f"no V2 tables in {tuning._INTERNODE_TABLE_DIR}"
    assert not list(tuning._INTERNODE_TABLE_DIR.glob("*.json"))
    for path in shipped_tables():
        match = NAME.match(path.name)
        assert match, path.name
        kernel_type, ep = match.group(3), int(match.group(4))
        rules = tuning.read_v2_table(path, kernel_type)
        assert rules, path.name
        assert all(r["num_experts"] % ep == 0 for r in rules), path.name


def test_shipped_tables_are_written_as_the_saver_writes(tmp_path):
    # A tuning save rewrites the whole file; it must not reorder or reformat
    # the rules it did not touch, so its diff is only the rules that changed.
    for path in shipped_tables():
        kernel_type = NAME.match(path.name).group(3)
        copy = tmp_path / path.name
        tuning.write_v2_table(
            copy, tuning.read_v2_table(path, kernel_type), kernel_type
        )
        assert copy.read_text() == path.read_text(), path.name


# -- format ------------------------------------------------------------------


def rule(phase="dispatch", **fields):
    values = dict(
        num_experts=64,
        topk=4,
        phase=phase,
        dtype="bf16",
        hidden_dim=1024,
        num_tokens=16,
        block_num=80,
        rdma_block_num=40,
        warp_per_block=4,
    )
    if phase == "combine":
        values.update(zero_copy=False, quant_type="none")
    values.update(fields)
    return values


def test_round_trip(tmp_path):
    rules = [
        rule(),
        rule(num_tokens=128, active_qps=1, avg_latency_us=75.12, bandwidth_gbps=2.0),
        rule("combine", active_qps=4, combine_load_first=True),
        rule("combine", num_tokens=128, combine_load_first=False),
        rule(num_experts=128, bandwidth_metric="grand_mean"),
    ]
    path = tmp_path / "t.csv"
    tuning.write_v2_table(path, reversed(rules), "InterNodeV2")
    assert list(tuning.read_v2_table(path, "InterNodeV2")) == sorted(
        rules, key=tuning._table_key
    )
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(tuning._TABLE_COLUMNS)
    # Sorted by selector; booleans 0/1; absent fields empty; 2.0 written as 2.
    assert lines[1] == "64,4,combine,bf16,1024,0,none,16,80,40,4,4,1,,,,,,"
    assert lines[3] == "64,4,dispatch,bf16,1024,,,16,80,40,4,,,,,,,,"
    assert lines[4] == "64,4,dispatch,bf16,1024,,,128,80,40,4,1,,75.12,2,,,,"


def test_numpy_metrics_are_written_as_numbers(tmp_path):
    np = pytest.importorskip("numpy")
    path = tmp_path / "t.csv"
    rules = [rule(avg_latency_us=np.float64(1.25), bandwidth_gbps=np.float32(2.5))]
    tuning.write_v2_table(path, rules, "InterNodeV2")
    (back,) = tuning.read_v2_table(path, "InterNodeV2")
    assert (back["avg_latency_us"], back["bandwidth_gbps"]) == (1.25, 2.5)


HEADER = ",".join(tuning._TABLE_COLUMNS)
COMBINE = dict(phase="combine", zero_copy="0", quant_type="none")


def row(**cells):
    values = dict(
        num_experts="64",
        topk="4",
        phase="dispatch",
        dtype="bf16",
        hidden_dim="1024",
        num_tokens="16",
        block_num="80",
        rdma_block_num="40",
        warp_per_block="4",
    )
    values.update(cells)
    return ",".join(values.get(name, "") for name in tuning._TABLE_COLUMNS)


@pytest.mark.parametrize(
    "kernel_type,lines,message",
    [
        ("InterNodeV2", [HEADER.replace(",active_qps", ""), row()], "columns must be"),
        ("InterNodeV2", [HEADER + ",note", row() + ","], "columns must be"),
        ("InterNodeV2", [HEADER, row() + ",x"], "more cells than columns"),
        ("InterNodeV2", [HEADER, row().rsplit(",", 3)[0]], "fewer cells than columns"),
        ("InterNodeV2", [HEADER, row(warp_per_block="")], "missing warp_per_block"),
        ("InterNodeV2", [HEADER, row(block_num="8.0")], "block_num='8.0'"),
        ("InterNodeV2", [HEADER, row(active_qps="0")], "active_qps=0 is below 1"),
        ("InterNodeV2", [HEADER, row(dtype="bf61")], "unknown dtype"),
        ("InterNodeV2", [HEADER, row(phase="both")], "neither dispatch nor combine"),
        ("InterNodeV2", [HEADER, row(zero_copy="0")], "zero_copy is combine-only"),
        (
            "InterNodeV2",
            [HEADER, row(phase="combine", zero_copy="0")],
            "missing quant_type",
        ),
        (
            "InterNodeV2",
            [HEADER, row(**COMBINE, combine_load_first="true")],
            "want 0 or 1",
        ),
        (
            "InterNodeV2LL",
            [HEADER, row(**COMBINE, combine_load_first="1")],
            "InterNodeV2 only",
        ),
        ("IntraNodeV2", [HEADER, row(active_qps="2")], "internode only"),
        (
            "InterNodeV2",
            [HEADER, row(), row(block_num="64")],
            "same selector as line 2",
        ),
    ],
)
def test_malformed_table_raises(tmp_path, kernel_type, lines, message):
    path = tmp_path / "t.csv"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match=message):
        tuning.read_v2_table(path, kernel_type)


def test_writer_refuses_what_the_reader_would(tmp_path):
    path = tmp_path / "t.csv"
    with pytest.raises(ValueError, match="no column for"):
        tuning.write_v2_table(path, [rule(note="x")], "InterNodeV2")
    with pytest.raises(ValueError, match="appears twice"):
        tuning.write_v2_table(path, [rule(), rule(block_num=64)], "InterNodeV2")
    with pytest.raises(ValueError, match="InterNodeV2 only"):
        tuning.write_v2_table(
            path, [rule("combine", combine_load_first=True)], "InterNodeV2LL"
        )
    assert not path.exists()


# -- save --------------------------------------------------------------------


@pytest.fixture
def table_dir(tmp_path, monkeypatch):
    """Point lookup and save at a temp directory as a fake 80-CU model."""
    monkeypatch.setenv("MORI_EP_V2_TUNING_DIR", str(tmp_path))
    monkeypatch.setattr(tuning._gpu, "detect_model", lambda: MODEL)
    monkeypatch.setattr(tuning._gpu, "arch_name", lambda: ARCH)
    monkeypatch.setattr(tuning._gpu, "cu_count", lambda: 80)
    tuning._V2_RULES_CACHE.clear()
    yield tmp_path
    tuning._V2_RULES_CACHE.clear()


def entry(**fields):
    values = dict(
        dtype="bf16",
        num_tokens=16,
        hidden_dim=1024,
        topk=4,
        experts_per_rank=4,
        block_num=80,
        rdma_block_num=40,
        warp_per_block=4,
    )
    values.update(fields)
    return values


def test_save_merges_by_selector(table_dir):
    path = tuning.save_internode_result(16, "dispatch", entry(active_qps=1))
    assert path == table_dir / tuning.v2_table_filename(ARCH, MODEL, "InterNodeV2", 16)
    tuning.save_internode_result(
        16, "combine", entry(active_qps=4, combine_load_first=True)
    )
    # Another expert count goes in the same file as its own rule.
    tuning.save_internode_result(16, "dispatch", entry(experts_per_rank=8))
    # The same selector replaces the rule.
    tuning.save_internode_result(16, "dispatch", entry(active_qps=2, block_num=64))
    rules = tuning.read_v2_table(path, "InterNodeV2")
    assert [
        (r["phase"], r["num_experts"], r["block_num"], r.get("active_qps"))
        for r in rules
    ] == [
        ("combine", 64, 80, 4),
        ("dispatch", 64, 64, 2),
        ("dispatch", 128, 80, None),
    ]
    # Lookup sees the saved rules, and each expert count only its own.
    assert tuning.internode_buckets(
        16, 1024, 4, torch.bfloat16, experts_per_rank=4
    ) == [(16, (64, 40, 4, 2), (80, 40, 4, 4), True)]
    assert tuning.internode_buckets(
        16, 1024, 4, torch.bfloat16, experts_per_rank=8
    ) == [(16, (80, 40, 4, None), None, False)]


def test_save_rejects_a_bad_rule_and_keeps_the_table(table_dir):
    path = tuning.save_internode_result(16, "dispatch", entry())
    before = path.read_text()
    with pytest.raises(ValueError, match="active_qps=0 is below 1"):
        tuning.save_internode_result(16, "dispatch", entry(active_qps=0, num_tokens=32))
    with pytest.raises(ValueError, match="internode only"):
        tuning.save_internode_result(
            8, "dispatch", entry(active_qps=2), kernel_family="intra"
        )
    assert path.read_text() == before


def test_schedule_description_names_the_table(table_dir):
    # Logged at op construction on rank 0, so a stale call here breaks the build.
    tuning.save_internode_result(16, "dispatch", entry(active_qps=1))
    cfg = EpDispatchCombineConfig(
        rank=0,
        world_size=16,
        gpu_per_node=8,
        hidden_dim=1024,
        max_num_inp_token_per_rank=128,
        num_experts_per_rank=4,
        num_experts_per_token=4,
        kernel_backend="hip",
        internode_kernel="v2",
    )
    text = tuning.describe_schedule(cfg, tuning.resolve_schedule(cfg))
    table = table_dir / tuning.v2_table_filename(ARCH, MODEL, "InterNodeV2", 16)
    assert f"found: {table}" in text
    # One rule, so its bucket and the terminal one are the same entry.
    assert "tokens    rest: dispatch v2 B/R/W=80/40/4 qps=1" in text
