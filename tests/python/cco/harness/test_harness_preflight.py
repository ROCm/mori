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
"""benchmark/cco/harness/preflight.py on the CPU: parsers on canned tool output, and the rules."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[4] / "benchmark" / "cco" / "harness"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"cco_harness_{name}", HARNESS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight = _load("preflight")


def _agent(n, name, kind, marketing):
    return f"""*******
Agent {n}
*******
  Name:                    {name}
  Uuid:                    {kind}-XX
  Marketing Name:          {marketing}
  Vendor Name:             AMD
  Device Type:             {kind}
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--{name}:sramecc+:xnack-
"""


ROCMINFO = (
    "ROCk module version 6.12.12 is loaded\n"
    + _agent(
        1, "AMD EPYC 9575F 64-Core Processor", "CPU", "AMD EPYC 9575F 64-Core Processor"
    )
    + _agent(2, "gfx950", "GPU", "AMD Instinct MI355X")
    + _agent(3, "gfx950", "GPU", "AMD Instinct MI355X")
)
IDLE = """============================ ROCm System Management Interface ============================
================================== KFD Processes ==================================
No KFD PIDs currently running
==================================================================================
"""
BUSY = """================================ KFD Processes =================================
KFD process information:
PID     PROCESS NAME  GPU(s)  VRAM USED   SDMA USED  CU OCCUPANCY
41213   python3       8       1234567890  0          0
=================================================================================
"""
TOPO = """=============================== Link Type between two GPUs ===============================
       GPU0         GPU1         GPU2
GPU0   0            XGMI         XGMI
GPU1   XGMI         0            PCIE
GPU2   XGMI         PCIE         0
================================== End of ROCm SMI Log ===================================
"""


def good_info(**overrides):
    info = {
        "host": "node",
        "env": {"MORI_ENABLE_SDMA": "1"},
        "repo": {
            "path": "/src/mori",
            "commit": "a" * 40,
            "branch": "b",
            "dirty": False,
        },
        "mori": {
            "path": "/src/mori/python/mori",
            "version": "1.0",
            "build_cco_sdma": True,
        },
        "rocm": {"version": "10.1.0", "source": "x"},
        "gpus": [{"name": "gfx950", "marketing_name": "AMD Instinct MI355X"}] * 8,
        "gpu_processes": [],
        "links": {"GPU0-GPU1": "XGMI", "GPU1-GPU0": "XGMI"},
        "dmabuf_move_notify": {"value": "y", "source": "/boot/config"},
    }
    info.update(overrides)
    return info


def levels(findings):
    return {f["code"]: f["level"] for f in findings}


def test_parsers_on_tool_output():
    assert preflight.parse_rocminfo(ROCMINFO) == [
        {"name": "gfx950", "marketing_name": "AMD Instinct MI355X"},
        {"name": "gfx950", "marketing_name": "AMD Instinct MI355X"},
    ]
    assert preflight.parse_showpids(IDLE) == []
    assert preflight.parse_showpids(BUSY) == [{"pid": 41213, "name": "python3"}]
    assert preflight.parse_showpids("garbage") is None
    links = preflight.parse_topotype(TOPO)
    assert (
        links["GPU0-GPU1"] == "XGMI"
        and links["GPU1-GPU2"] == "PCIE"
        and len(links) == 6
    )
    assert preflight.parse_build_flags("# generated\nBUILD_CCO_SDMA = True\n") is True
    assert preflight.parse_build_flags("BUILD_CCO_SDMA = False\n") is False
    assert preflight.parse_build_flags("") is None
    assert preflight.parse_kconfig("CONFIG_A=y\nCONFIG_DMABUF_MOVE_NOTIFY=y\n") == "y"
    assert preflight.parse_kconfig("# CONFIG_DMABUF_MOVE_NOTIFY is not set\n") == "n"
    assert preflight.parse_kconfig("CONFIG_A=y\n") is None


def test_a_good_environment_passes():
    assert levels(preflight.evaluate(good_info())) == {}


@pytest.mark.parametrize(
    "overrides, code, level",
    [
        ({"gpus": [{"name": "gfx1250", "marketing_name": "x"}]}, "arch", "FAIL"),
        ({"gpus": None}, "gpus", "FAIL"),
        ({"gpu_processes": [{"pid": 1, "name": "python3"}]}, "busy", "FAIL"),
        ({"gpu_processes": None}, "busy", "FAIL"),
        (
            {"mori": {"path": None, "version": None, "build_cco_sdma": None}},
            "mori",
            "FAIL",
        ),
        (
            {"mori": {"path": "/src/mori/python/mori", "build_cco_sdma": False}},
            "sdma-build",
            "FAIL",
        ),
        ({"env": {}}, "sdma-env", "FAIL"),
        ({"rocm": {"version": "7.2.26015", "source": "x"}}, "rocm-7.2", "WARN"),
        (
            {"dmabuf_move_notify": {"value": None, "source": None}},
            "dmabuf-move-notify",
            "WARN",
        ),
        ({"links": {"GPU0-GPU1": "PCIE"}}, "links", "WARN"),
        (
            {
                "mori": {
                    "path": "/sgl-workspace/mori/python/mori",
                    "build_cco_sdma": True,
                }
            },
            "mori-source",
            "WARN",
        ),
        (
            {"repo": {"path": "/src/mori", "commit": "a" * 40, "dirty": True}},
            "repo-dirty",
            "WARN",
        ),
    ],
)
def test_rules(overrides, code, level):
    assert levels(preflight.evaluate(good_info(**overrides))).get(code) == level


def test_overrides_relax_the_rules():
    busy = good_info(gpu_processes=[{"pid": 1, "name": "x"}])
    assert levels(preflight.evaluate(busy, allow_busy=True))["busy"] == "WARN"
    unbuilt = good_info(env={"MORI_ENABLE_SDMA": "1", "BUILD_CCO_SDMA": "ON"})
    unbuilt["mori"] = {"path": "/src/mori/python/mori", "build_cco_sdma": False}
    assert "sdma-build" not in levels(preflight.evaluate(unbuilt))
    assert levels(preflight.evaluate(good_info(env={}), require_sdma=False)) == {}
    assert (
        levels(
            preflight.evaluate(good_info(gpus=[{"name": "gfx1250"}]), require_arch="")
        )
        == {}
    )


def test_collect_reads_the_tools_through_the_runner():
    outputs = {
        ("rocminfo",): (0, ROCMINFO),
        ("rocm-smi", "--showpids"): (0, IDLE),
        ("rocm-smi", "--showtopotype"): (0, TOPO),
        ("amd-smi", "version"): (
            0,
            "AMDSMI Tool: 26.0 | AMDSMI Library version: 26.0 | ROCm version: 10.1.0\n",
        ),
    }

    def fake(cmd, timeout=60):
        return outputs.get(tuple(cmd), (None, "not found"))

    info = preflight.collect(run=fake)
    assert [g["name"] for g in info["gpus"]] == ["gfx950", "gfx950"]
    assert info["gpu_processes"] == []
    assert info["links"]["GPU2-GPU1"] == "PCIE"
    assert "repo" not in info
    json.dumps(info)  # the record must serialize


def test_main_writes_the_record_and_exits_on_a_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "collect",
        lambda repo=None: good_info(gpu_processes=[{"pid": 7, "name": "x"}]),
    )
    out = tmp_path / "env.json"
    assert preflight.main(["--out", str(out)]) == 1
    record = json.loads(out.read_text())
    assert record["result"] == "FAIL" and record["findings"][0]["code"] == "busy"
    assert preflight.main(["--out", str(out), "--allow-busy"]) == 0


def test_git_queries_allow_a_checkout_owned_by_someone_else(monkeypatch, tmp_path):
    seen = []

    def fake(cmd, timeout=60):
        seen.append(cmd)
        return 0, "abc\n"

    monkeypatch.setattr(preflight, "sh", fake)
    assert preflight._git(tmp_path, "rev-parse", "HEAD") == "abc"
    assert seen[0][:3] == ["git", "-c", f"safe.directory={tmp_path}"]
