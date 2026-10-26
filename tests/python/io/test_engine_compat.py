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
"""Tests for EngineDesc handshake compatibility fields (Gap 12)."""
import os
import pytest
from contextlib import contextmanager

from mori.io import (
    IOEngineConfig,
    BackendType,
    IOEngine,
    EngineDesc,
    RdmaBackendConfig,
    set_log_level,
)
from mori import cpp as mori_cpp


@contextmanager
def temporary_env(env_name: str, value: str):
    old_value = os.environ.get(env_name)
    os.environ[env_name] = value
    try:
        yield
    finally:
        if old_value is None:
            del os.environ[env_name]
        else:
            os.environ[env_name] = old_value


class TestEngineDescCompatFields:
    """Verify that the new compatibility fields are visible on EngineDesc."""

    def _create_engine(self, name):
        config = IOEngineConfig(host="127.0.0.1", port=0)
        engine = IOEngine(key=name, config=config)
        rdma_config = RdmaBackendConfig()
        with temporary_env("MORI_DISABLE_AUTO_XGMI", "1"):
            engine.create_backend(BackendType.RDMA, rdma_config)
        return engine

    def test_protocol_version_field_exists(self):
        engine = self._create_engine("compat_test_1")
        desc = engine.get_engine_desc()
        assert hasattr(desc, "protocol_version")
        assert desc.protocol_version == mori_cpp.MORI_IO_PROTOCOL_VERSION
        assert desc.protocol_version >= 1

    def test_agent_mode_field_exists(self):
        engine = self._create_engine("compat_test_2")
        desc = engine.get_engine_desc()
        assert hasattr(desc, "agent_mode")
        assert desc.agent_mode == int(mori_cpp.AgentMode.READ_WRITE)

    def test_block_size_field_exists(self):
        engine = self._create_engine("compat_test_3")
        desc = engine.get_engine_desc()
        assert hasattr(desc, "block_size")
        assert desc.block_size == 0  # default, not set by IOEngine

    def test_msgpack_roundtrip_preserves_fields(self):
        engine = self._create_engine("compat_test_4")
        desc = engine.get_engine_desc()

        packed = desc.pack()
        unpacked = EngineDesc.unpack(packed)

        assert unpacked.protocol_version == desc.protocol_version
        assert unpacked.agent_mode == desc.agent_mode
        assert unpacked.block_size == desc.block_size
        assert unpacked == desc

    def test_agent_mode_enum_values(self):
        assert int(mori_cpp.AgentMode.NONE) == 0
        assert int(mori_cpp.AgentMode.READ) == 1
        assert int(mori_cpp.AgentMode.WRITE) == 2
        assert int(mori_cpp.AgentMode.READ_WRITE) == 3

    def test_protocol_version_constant(self):
        assert mori_cpp.MORI_IO_PROTOCOL_VERSION >= 1
