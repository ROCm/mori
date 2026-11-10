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
import pytest

umbp = pytest.importorskip("mori.umbp")


def test_config_constructor_reads_environment(monkeypatch):
    monkeypatch.setenv("UMBP_SIM_ENABLE", "1")
    monkeypatch.setenv("UMBP_SIM_PROFILE", "/tmp/sim-profile.json")
    monkeypatch.setenv("UMBP_SSD_SINGLE_FLIGHT", "0")
    cfg = umbp.UMBPConfig()
    assert cfg.sim.enabled
    assert cfg.sim.profile_path == "/tmp/sim-profile.json"
    assert not cfg.ssd.single_flight_reads
